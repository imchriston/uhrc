"""
train_uhrc.py  —  UHRC Hierarchical Behavioural Cloning
════════════════════════════════════════════════════════════════════════════════
Uses uhrc_fixed architecture where:
  - H-level subgoal is injected into L-level BEFORE the final attention pass
  - Carry is a rolling buffer (UHRCCarry) not a flat tensor
  - Model takes [B, T, 49] for training — processes each timestep sequentially
    so the carry rolls correctly through time
  - SEQ_LEN controls gradient window width, NOT temporal memory
    (temporal memory comes from carry_len=16 in the architecture)

Loss terms:
  l_act    Weighted MSE on [Fz, τx, τy, τz]
  l_sub    Weighted MSE on H-level velocity subgoal [vx, vy, vz]
  l_track  Cosine alignment — τy↔vx, τx↔vy (correct quadrotor physics)
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, cast

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.hrm.hrm_act_v1 import UHRC, UHRC_Config, UHRCCarry
from uhrc_dataset_h import ControlDataset


# ══════════════════════════════════════════════════════════════════════════════
#  ███  TUNING  ███
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "data/Astardata.npz"

# SEQ_LEN=8 trains the carry across 8 consecutive steps per gradient update —
# the carry actually learns to accumulate goal memory over time.
# With BATCH_SIZE=128 this keeps VRAM safe on 6GB.
SEQ_LEN      =1
STRIDE       = 1         # all windows — with carry arch each window is independent
VAL_FRAC     = 0.1
MAX_EPISODES = None      # use ALL generated data — was 2400, caused 7× overfit gap

EPOCHS        = 60
WARMUP_EPOCHS = 5        # longer warmup — 2 was too abrupt for cosine over 60 epochs
BATCH_SIZE    = 512      # halved from 256 to accommodate SEQ_LEN=8
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 3e-3     # increased from 1e-3 — reduces overfit on smaller datasets
GRAD_CLIP     = 1.0

# Action weights ∝ 1/std²
ACTION_WEIGHTS  = [0.68, 71.4, 69.0, 50.0]   # [Fz, τx, τy, τz]
ALPHA_SUBGOAL   = 5.0
SUBGOAL_WEIGHTS = [1.5, 1.5, 0.1]             # [vx, vy, vz] — vz down-weighted
# Tracking loss starts near zero (subgoals near-zero early on) then grows.
# Reduce from 5.0 to 2.0 — was wasting gradient budget in early epochs.
ALPHA_TRACKING  = 2.0

# Architecture — must match uhrc_fixed.py defaults
HIDDEN_SIZE  = 256
CARRY_LEN    = 16
EXPANSION    = 4.0
NUM_HEADS    = 4
H_CYCLES     = 2
L_CYCLES     = 2
H_LAYERS     = 2
L_LAYERS     = 2
HOVER_THRUST = 9.81
DETACH_CARRY = True

USE_AMP          = True
NUM_WORKERS      = 4
COMPILE_MODEL    = False
EVAL_EVERY       = 5
RESUME_FROM      = None    # "checkpoints/uhrc_latest_slam.pth" to resume
SAVE_DIR         = "checkpoints"
SEED             = 42

# ══════════════════════════════════════════════════════════════════════════════
#  END TUNING
# ══════════════════════════════════════════════════════════════════════════════


def episode_split(dataset, ep_ids: np.ndarray, val_frac: float):
    rng        = np.random.default_rng(SEED)
    unique_eps = np.unique(ep_ids)
    rng.shuffle(unique_eps)
    n_val     = max(1, int(len(unique_eps) * val_frac))
    val_eps   = set(unique_eps[:n_val].tolist())
    train_eps = set(unique_eps[n_val:].tolist())
    train_idx = [i for i, ep in enumerate(ep_ids) if ep in train_eps]
    val_idx   = [i for i, ep in enumerate(ep_ids) if ep in val_eps]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_raw_state_dict(model: nn.Module) -> dict:
    """Unwrap torch.compile OptimizedModule if present."""
    inner = getattr(model, "_orig_mod", model)
    return nn.Module.state_dict(inner)   # type: ignore[arg-type]


def get_lr(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_action_weights(device: torch.device) -> torch.Tensor:
    w = torch.tensor(ACTION_WEIGHTS, dtype=torch.float32, device=device)
    return w / w.mean()


def build_subgoal_weights(device: torch.device) -> torch.Tensor:
    w = torch.tensor(SUBGOAL_WEIGHTS, dtype=torch.float32, device=device)
    return w / w.mean()


def weighted_action_loss(pred: torch.Tensor, target: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weights).sum(dim=-1).mean()


def weighted_subgoal_loss(pred: torch.Tensor, target: torch.Tensor,
                          weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weights).sum(dim=-1).mean()


def tracking_loss(pred_act: torch.Tensor,
                  pred_sub: torch.Tensor) -> torch.Tensor:
    """
    Cosine alignment between H-level subgoal direction and L-level torques.

    Correct quadrotor coupling:
      τy (pitch) → forward acceleration → vx
      τx (roll)  → lateral acceleration → vy

    Penalises L when torques oppose H's commanded direction.
    Handles both [B,4]/[B,3] and [B,T,4]/[B,T,3].
    """
    if pred_act.dim() == 3:
        B, T, _ = pred_act.shape
        pred_act = pred_act.reshape(B * T, -1)
        pred_sub = pred_sub.reshape(B * T, -1)

    vx = pred_sub[:, 0]
    vy = pred_sub[:, 1]
    ty = pred_act[:, 2]   # pitch → vx
    tx = pred_act[:, 1]   # roll  → vy

    sub_vec    = torch.stack([vx, vy], dim=-1)
    torque_vec = torch.stack([ty, tx], dim=-1)

    sub_n    = sub_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    torque_n = torque_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    alignment = (sub_vec / sub_n * torque_vec / torque_n).sum(dim=-1)
    return torch.clamp(-alignment, min=0.0).mean()


def run_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   Optional[optim.Optimizer],
    act_weights: torch.Tensor,
    sub_weights: torch.Tensor,
    device:      torch.device,
    is_train:    bool,
    scaler:      Optional[torch.cuda.amp.GradScaler] = None,
    pbar:        Optional[tqdm] = None,
) -> dict:
    model.train(is_train)
    total_loss = total_act = total_sub = total_track = 0.0
    n_batches  = len(loader)
    use_amp    = scaler is not None and device.type == "cuda"

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            state   = batch["state"].to(device)    # [B, T, 45]
            tgt_act = batch["action"].to(device)   # [B, T, 4]
            tgt_sub = batch["subgoal"].to(device)  # [B, T, 3]

            with torch.autocast("cuda", enabled=use_amp):
                # uhrc_fixed rolls carry across T steps internally.
                # Each step sees the carry built from all previous steps.
                # carry=None → auto-init from learned H_init/L_init buffers.
                pred_act_seq, pred_sub_seq, _ = model(state, carry=None)
                # pred_act_seq: [B, T, 4]
                # pred_sub_seq: [B, T, 3]

                B, T, _ = pred_act_seq.shape
                pa_flat = pred_act_seq.reshape(B * T, -1)
                ps_flat = pred_sub_seq.reshape(B * T, -1)
                ta_flat = tgt_act.reshape(B * T, -1)
                ts_flat = tgt_sub.reshape(B * T, -1)

                l_act   = weighted_action_loss(pa_flat, ta_flat, act_weights)
                l_sub   = weighted_subgoal_loss(ps_flat, ts_flat, sub_weights)
                l_track = tracking_loss(pred_act_seq, pred_sub_seq)
                loss    = l_act + ALPHA_SUBGOAL * l_sub + ALPHA_TRACKING * l_track

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

            total_loss  += loss.item()
            total_act   += l_act.item()
            total_sub   += l_sub.item()
            total_track += l_track.item()

            if pbar is not None:
                pbar.update(1)

    n = max(n_batches, 1)
    return {"loss": total_loss/n, "action": total_act/n,
            "subgoal": total_sub/n, "tracking": total_track/n}


def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUHRC Training (uhrc_fixed) — device={device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print(f"[ERROR] Data not found: {data_path}")
        sys.exit(1)

    print(f"Loading: {data_path}")
    raw_dataset = ControlDataset(str(data_path), seq_len=SEQ_LEN, normalize=True)
    ep_ids      = raw_dataset.episode_id_per_window
    print(f"  Episodes: {len(np.unique(ep_ids)):,}  "
          f"Windows: {len(raw_dataset):,}")

    # Stride — subsample windows to reduce epoch length
    if STRIDE > 1:
        keep_idx: list[int] = []
        for ep in np.unique(ep_ids):
            ep_mask = np.where(ep_ids == ep)[0]
            keep_idx.extend(ep_mask[::STRIDE].tolist())
        keep_idx.sort()
    else:
        keep_idx = list(range(len(raw_dataset)))

    dataset        = Subset(raw_dataset, keep_idx)
    strided_ep_ids = np.array([ep_ids[i] for i in keep_idx], dtype=np.int32)

    # Episode cap
    if MAX_EPISODES is not None:
        unique_eps = np.unique(strided_ep_ids)
        if len(unique_eps) > MAX_EPISODES:
            rng    = np.random.default_rng(SEED)
            chosen = set(
                rng.choice(unique_eps, size=MAX_EPISODES, replace=False).tolist()
            )
            cap_idx        = [i for i, ep in enumerate(strided_ep_ids)
                               if ep in chosen]
            dataset        = Subset(raw_dataset, [keep_idx[i] for i in cap_idx])
            strided_ep_ids = strided_ep_ids[cap_idx]
            print(f"  Capped to {MAX_EPISODES:,} episodes "
                  f"({len(cap_idx):,} windows)")

    train_data, val_data = episode_split(dataset, strided_ep_ids, VAL_FRAC)
    print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}")

    # Save norm stats
    norm_path = os.path.join(SAVE_DIR, "norm_statsA*.npz")
    np.savez(norm_path,
             obs_mean=raw_dataset.obs_mean, obs_std=raw_dataset.obs_std,
             act_mean=raw_dataset.act_mean, act_std=raw_dataset.act_std,
             sub_mean=raw_dataset.sub_mean, sub_std=raw_dataset.sub_std)
    print(f"  Norm stats → {norm_path}")

    _wkw: dict = dict(num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=(NUM_WORKERS > 0))
    train_loader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True, **_wkw,
    )
    val_loader = DataLoader(
        dataset=val_data, batch_size=BATCH_SIZE,
        shuffle=False, **_wkw,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    config = UHRC_Config(
        hidden_size  = HIDDEN_SIZE,
        carry_len    = CARRY_LEN,
        expansion    = EXPANSION,
        num_heads    = NUM_HEADS,
        H_cycles     = H_CYCLES,
        L_cycles     = L_CYCLES,
        H_layers     = H_LAYERS,
        L_layers     = L_LAYERS,
        hover_thrust = HOVER_THRUST,
        detach_carry = DETACH_CARRY,
    )
    model    = UHRC(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nUHRC — {n_params:,} parameters")
    print(f"  Architecture: H_cycles={H_CYCLES} L_cycles={L_CYCLES} "
          f"carry_len={CARRY_LEN} hidden={HIDDEN_SIZE}")
    print(f"  Subgoal injected into L BEFORE final attention pass ✓")

    optimizer   = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY)
    act_weights = build_action_weights(device)
    sub_weights = build_subgoal_weights(device)
    use_amp     = USE_AMP and device.type == "cuda"
    scaler      = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    if RESUME_FROM and os.path.exists(str(RESUME_FROM)):
        ckpt = torch.load(str(RESUME_FROM), map_location=device)
        # Load model weights — strict=False tolerates minor arch changes
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  ⚠️  {len(missing)} layers not in checkpoint (random init)")
        if unexpected:
            print(f"  ⚠️  {len(unexpected)} unexpected keys ignored")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, "
              f"best val={best_val_loss:.4f}")
    else:
        print("  Training from scratch")

    # torch.compile (PyTorch ≥ 2.0)
    if COMPILE_MODEL and hasattr(torch, "compile"):
        print("  Compiling model...")
        model = cast(nn.Module, torch.compile(model))
        print("  Done.")

    print(f"  AMP: {'ON' if use_amp else 'OFF'}  "
          f"batch={BATCH_SIZE}  SEQ_LEN={SEQ_LEN}  STRIDE={STRIDE}")

    # ── Training loop ─────────────────────────────────────────────────────────
    hdr = (f"\n{'Ep':>5}  {'T-Loss':>8}  {'T-Act':>8}  {'T-Sub':>8}  "
           f"{'T-Trk':>8}  {'V-Loss':>8}  {'V-Act':>8}  {'V-Sub':>8}  "
           f"{'LR':>8}  {'s':>6}")
    print(hdr)
    print("─" * len(hdr))

    epoch_bar = tqdm(total=len(train_loader), unit="it", dynamic_ncols=True)

    for epoch in range(start_epoch, EPOCHS):
        t0     = time.time()
        do_eval = ((epoch + 1) % EVAL_EVERY == 0) or (epoch + 1 == EPOCHS)
        n_this  = len(train_loader) + (len(val_loader) if do_eval else 0)
        epoch_bar.reset(total=n_this)
        epoch_bar.set_description(f"Epoch {epoch+1:>3}/{EPOCHS}")

        lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_m = run_epoch(model, train_loader, optimizer,
                            act_weights, sub_weights, device,
                            is_train=True, scaler=scaler, pbar=epoch_bar)
        torch.cuda.empty_cache()

        if do_eval:
            val_m    = run_epoch(model, val_loader, None,
                                 act_weights, sub_weights, device,
                                 is_train=False, scaler=None, pbar=epoch_bar)
            improved = val_m["loss"] < best_val_loss

            if improved:
                best_val_loss = val_m["loss"]
                torch.save(get_raw_state_dict(model),
                           os.path.join(SAVE_DIR, "uhrc_bestA*.pth"))

            raw_sd = get_raw_state_dict(model)
            torch.save(
                {"epoch": epoch + 1, "model": raw_sd,
                 "optimizer": optimizer.state_dict(),
                 "config": config.model_dump(),
                 "val_loss": val_m["loss"]},
                os.path.join(SAVE_DIR, "uhrc_latestA*.pth"),
            )

            elapsed = time.time() - t0
            marker  = "  ✓" if improved else ""
            tqdm.write(
                f"{epoch+1:>5}  "
                f"{train_m['loss']:>8.4f}  {train_m['action']:>8.4f}  "
                f"{train_m['subgoal']:>8.4f}  {train_m['tracking']:>8.4f}  "
                f"{val_m['loss']:>8.4f}  {val_m['action']:>8.4f}  "
                f"{val_m['subgoal']:>8.4f}  "
                f"{lr:>8.2e}  {elapsed:>6.1f}" + marker
            )
            epoch_bar.set_postfix(
                val=f"{val_m['loss']:.4f}",
                best=f"{best_val_loss:.4f}",
                lr=f"{lr:.1e}",
            )
        else:
            elapsed = time.time() - t0
            tqdm.write(
                f"{epoch+1:>5}  "
                f"{train_m['loss']:>8.4f}  {train_m['action']:>8.4f}  "
                f"{train_m['subgoal']:>8.4f}  {train_m['tracking']:>8.4f}  "
                f"{'---':>8}  {'---':>8}  {'---':>8}  "
                f"{lr:>8.2e}  {elapsed:>6.1f}"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    epoch_bar.close()
    print(f"\nDone.  Best val: {best_val_loss:.4f}")
    print(f"  Best    : {SAVE_DIR}/uhrc_bestA*.pth")
    print(f"  Latest  : {SAVE_DIR}/uhrc_latestA*.pth\n")


if __name__ == "__main__":
    train()