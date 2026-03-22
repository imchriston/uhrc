"""
train_td3.py — TD3 fine-tuning of BC-pretrained UHRC
═════════════════════════════════════════════════════
Actor  : uhrc_fixed.UHRC  (BC warm-start, single-step [B,45], carry-based)
Critic : TD3Critic         Q(obs[B,45], action[B,4]) — twin MLP + lidar encoder
                           + LayerNorm to prevent dead-critic collapse
Env    : ForestEnv         45-dim obs

Key fixes over previous version:
  - Import from uhrc_fixed not hrm_act_v1
  - terminal vs done: timeout is NOT terminal, bootstraps normally
  - HER normalisation: raw=0 → -mean/std, NOT 0.0 in normalised space
  - Action scale normalisation in critic (Fz/10, τ/0.3)
  - LayerNorm in critic to prevent collapse on homogeneous buffer
  - Buffer frozen detection + zero-obs guard
  - frozen_bc anchor: BC penalty in actor loss prevents catastrophic forgetting
  - Q_TARGET_CLIP=3.0 — realistic max Q after REWARD_SCALE=0.01
  - Qmean guard is temporary (QMEAN_FREEZE_LIMIT), not permanent
"""
from __future__ import annotations

import copy
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.hrm.hrm_act_v1 import UHRC, UHRC_Config, UHRCCarry
from uhrc_env   import ForestEnv



TOTAL_TIMESTEPS         = 1_000_000
BUFFER_SIZE             = 300_000
BATCH_SIZE              = 256
WARMUP_STEPS            = 5_000
UPDATES_PER_STEP        = 2         # train critic faster on current buffer

LR_ACTOR                = 3e-5      # slightly higher — critic is reliable now
LR_CRITIC               = 3e-4

GAMMA                   = 0.99
TAU                     = 0.005

EXPLORATION_NOISE_START = 0.20      # higher start — need more variance near BC policy
EXPLORATION_NOISE_END   = 0.05
EXPLORATION_DECAY_STEPS = 600_000   # slower decay — keep exploring longer

POLICY_NOISE            = 0.10
NOISE_CLIP              = 0.25
POLICY_DELAY            = 5

REWARD_SCALE            = 0.01
Q_TARGET_CLIP           = 3.0

# BC anchor — 1.0 lets Q gradient dominate 3:1 over BC penalty
# At 3.0 the actor was barely moving — Q at -0.16 means gradient ≈ 0.16,
# BC penalty ≈ 0.003, so 3×0.003=0.009 vs 0.16 means actor DID move but
# the Q surface is too flat to provide direction. Lower lambda frees exploration.
BC_LAMBDA               = 1.0

# Qmean guard — temporary freeze when critic overestimates with 0% success
QMEAN_FREEZE_LIMIT      = 3         # reload BC after this many consecutive freezes

BC_CHECKPOINT           = "checkpoints/uhrc_bestA*.pth"
STATS_PATH              = "checkpoints/norm_statsA*.npz"
SAVE_DIR                = "checkpoints"
LOG_EVERY               = 2_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════


def make_config() -> UHRC_Config:
    """Must match BC training exactly."""
    return UHRC_Config(
        hidden_size=256, carry_len=16, expansion=4.0, num_heads=4,
        H_cycles=2, L_cycles=2, H_layers=2, L_layers=2,
        hover_thrust=9.81, detach_carry=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  REPLAY BUFFER
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, cap: int, obs_dim: int, act_dim: int):
        self.cap       = cap
        self.obs       = np.zeros((cap, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((cap, act_dim), dtype=np.float32)
        self.rewards   = np.zeros(cap,            dtype=np.float32)
        self.next_obs  = np.zeros((cap, obs_dim), dtype=np.float32)
        self.terminals = np.zeros(cap,            dtype=np.float32)
        # terminal=1 only for crash/success — timeout is NOT terminal (bootstraps)
        self.ptr       = 0
        self.size      = 0

    def add(self, obs, action, reward, next_obs, terminal):
        obs_f32      = np.asarray(obs,      dtype=np.float32)
        next_obs_f32 = np.asarray(next_obs, dtype=np.float32)
        if not np.isfinite(obs_f32).all() or not np.isfinite(next_obs_f32).all():
            return
        self.obs[self.ptr]       = obs_f32
        self.actions[self.ptr]   = np.asarray(action, dtype=np.float32)
        self.rewards[self.ptr]   = float(reward) * REWARD_SCALE
        self.next_obs[self.ptr]  = next_obs_f32
        self.terminals[self.ptr] = float(terminal)
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int, device: str):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx],       device=device),
            torch.tensor(self.actions[idx],   device=device),
            torch.tensor(self.rewards[idx],   device=device),
            torch.tensor(self.next_obs[idx],  device=device),
            torch.tensor(self.terminals[idx], device=device),
        )

    def __len__(self): return self.size


# ──────────────────────────────────────────────────────────────────────────────
#  ACTOR
# ──────────────────────────────────────────────────────────────────────────────

class TD3Actor(nn.Module):
    """
    Wraps uhrc_fixed.UHRC as a TD3 deterministic actor.
    collect_action(): [1,45] + carry → action, subgoal, new_carry  (rollout)
    forward()       : [B,45], carry=None → action [B,4]             (updates)
    """
    def __init__(self, uhrc: UHRC):
        super().__init__()
        self.uhrc = uhrc

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        act, _, _ = self.uhrc(obs, carry=None)
        return act

    @torch.no_grad()
    def collect_action(
        self, obs: torch.Tensor, carry: UHRCCarry | None
    ) -> tuple[np.ndarray, np.ndarray, UHRCCarry | None]:
        act, sub, new_carry = self.uhrc(obs, carry=carry)
        return act[0].cpu().numpy(), sub[0].cpu().numpy(), new_carry


# ──────────────────────────────────────────────────────────────────────────────
#  CRITIC — Twin Q(obs, action)
# ──────────────────────────────────────────────────────────────────────────────

class _QBranch(nn.Module):
    SCALAR_DIM = 13   # obs[0:13]: v_B, w_B, g_B, goal_rel_B, goal_dist_norm
    LIDAR_DIM  = 32   # obs[13:45]
    ACT_DIM    = 4
    LIDAR_CH   = 16
    SCALAR_OUT = 128
    FUSED_DIM  = 128 + 16 * 32   # 640
    HIDDEN     = 256

    def __init__(self):
        super().__init__()
        self.lidar_enc = nn.Sequential(
            nn.Conv1d(1, self.LIDAR_CH, kernel_size=5, padding=2), nn.SiLU(),
            nn.Conv1d(self.LIDAR_CH, self.LIDAR_CH, kernel_size=3, padding=1), nn.SiLU(),
            nn.Flatten(start_dim=1),
        )
        self.scalar_enc = nn.Sequential(
            nn.Linear(self.SCALAR_DIM + self.ACT_DIM, self.SCALAR_OUT), nn.SiLU(),
            nn.LayerNorm(self.SCALAR_OUT),   # prevents dead-critic collapse
            nn.Linear(self.SCALAR_OUT, self.SCALAR_OUT), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.FUSED_DIM, self.HIDDEN), nn.SiLU(),
            nn.LayerNorm(self.HIDDEN),
            nn.Linear(self.HIDDEN, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)  # type: ignore[arg-type]
        self.head[-1].bias.data.zero_()                   # type: ignore[union-attr]

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Normalise actions to similar scale before linear layer
        # Fz ∈ [0,20] → /10 → [0,2]   τ ∈ [-0.5,0.5] → /0.3 → [-1.67,1.67]
        action_norm = torch.stack([
            action[:, 0] / 10.0,
            action[:, 1] / 0.3,
            action[:, 2] / 0.3,
            action[:, 3] / 0.3,
        ], dim=-1)
        scalar = obs[:, :self.SCALAR_DIM]
        lidar  = obs[:, self.SCALAR_DIM:].unsqueeze(1)
        s_emb  = self.scalar_enc(torch.cat([scalar, action_norm], dim=-1))
        l_emb  = self.lidar_enc(lidar)
        return self.head(torch.cat([s_emb, l_emb], dim=-1))


class TD3Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = _QBranch()
        self.q2 = _QBranch()

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.q1(obs, action), self.q2(obs, action)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(obs, action)


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def detach_carry(carry: UHRCCarry | None) -> UHRCCarry | None:
    if carry is None: return None
    return UHRCCarry(z_H=carry.z_H.detach(), z_L=carry.z_L.detach())


def actor_healthy(actor: TD3Actor) -> bool:
    for p in actor.parameters():
        if torch.isnan(p.data).any() or torch.isinf(p.data).any():
            return False
    return True


def exploration_noise(step: int) -> float:
    frac = min(step / EXPLORATION_DECAY_STEPS, 1.0)
    return EXPLORATION_NOISE_START + frac * (EXPLORATION_NOISE_END - EXPLORATION_NOISE_START)


def load_bc(path: str, config: UHRC_Config) -> UHRC:
    model    = UHRC(config).to(DEVICE)
    raw_sd   = torch.load(path, map_location=DEVICE, weights_only=True)
    # Handle checkpoints that wrap weights in a dict
    if isinstance(raw_sd, dict):
        for key in ("model", "state_dict", "actor", "net"):
            if key in raw_sd and isinstance(raw_sd[key], dict):
                raw_sd = raw_sd[key]; break
    model_sd = model.state_dict()
    compat   = {k: v for k, v in raw_sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    missing  = [k for k in model_sd if k not in compat]
    model.load_state_dict(compat, strict=False)
    if not missing:
        print("  ✅  BC checkpoint loaded cleanly")
    else:
        print(f"  ⚠️  {len(missing)} layers randomly initialised")
    return model


def reload_bc(actor: TD3Actor, actor_target: TD3Actor,
              config: UHRC_Config, actor_opt: optim.Optimizer):
    print("  ❌  Reloading BC checkpoint")
    uhrc = load_bc(BC_CHECKPOINT, config)
    actor.uhrc.load_state_dict(uhrc.state_dict())
    actor_target.uhrc.load_state_dict(uhrc.state_dict())
    actor_opt.state.clear()   # stale momentum is poisonous after collapse
    print("  ✅  BC restored")


# ──────────────────────────────────────────────────────────────────────────────
#  TD3 UPDATE
# ──────────────────────────────────────────────────────────────────────────────

def td3_update(
    actor:         TD3Actor,
    actor_target:  TD3Actor,
    critic:        TD3Critic,
    critic_target: TD3Critic,
    frozen_bc:     TD3Actor,        # frozen BC anchor — prevents forgetting
    actor_opt:     optim.Optimizer,
    critic_opt:    optim.Optimizer,
    buffer:        ReplayBuffer,
    update_count:  int,
) -> dict:
    obs, actions, rewards, next_obs, terminals = buffer.sample(BATCH_SIZE, DEVICE)

    # ── Critic update ─────────────────────────────────────────────────────────
    with torch.no_grad():
        noise        = (torch.randn_like(actions) * POLICY_NOISE).clamp(
                            -NOISE_CLIP, NOISE_CLIP)
        next_actions = actor_target(next_obs) + noise
        q1n, q2n     = critic_target(next_obs, next_actions)
        q_next       = torch.min(q1n, q2n).squeeze(-1)
        # timeout is NOT terminal — bootstrap through it (terminals flag)
        q_target     = (rewards + GAMMA * (1.0 - terminals) * q_next).clamp(
                            -Q_TARGET_CLIP, Q_TARGET_CLIP)

    q1, q2      = critic(obs, actions)
    critic_loss = F.mse_loss(q1.squeeze(-1), q_target) + \
                  F.mse_loss(q2.squeeze(-1), q_target)

    if not torch.isfinite(critic_loss):
        return {"critic_loss": 0., "actor_loss": 0., "q_target": 0.,
                "q_actor": 0., "bc_pen": 0.}

    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_opt.step()

    # ── Delayed actor update ──────────────────────────────────────────────────
    actor_loss_val = 0.0
    q_actor_val    = 0.0
    bc_pen_val     = 0.0
    if update_count % POLICY_DELAY == 0:
        pred       = actor(obs)
        q1_val     = critic.q1_only(obs, pred).mean()

        if not torch.isfinite(q1_val):
            return {"critic_loss": critic_loss.item(), "actor_loss": 0.,
                    "q_target": float(q_target.mean()), "q_actor": 0., "bc_pen": 0.}

        # BC anchor — penalise divergence from expert policy
        # Prevents catastrophic forgetting of BC navigation behaviour
        with torch.no_grad():
            bc_actions = frozen_bc(obs)
        bc_penalty = F.mse_loss(pred, bc_actions)

        actor_loss = -q1_val + BC_LAMBDA * bc_penalty
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.3)
        actor_opt.step()
        actor_loss_val = actor_loss.item()
        q_actor_val    = float(q1_val)
        bc_pen_val     = float(bc_penalty)

        soft_update(actor_target, actor, TAU)
        soft_update(critic_target, critic, TAU)

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss":  actor_loss_val,
        "q_target":    float(q_target.mean()),
        "q_actor":     q_actor_val,
        "bc_pen":      bc_pen_val,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def train():
    print(f"\nTD3 — UHRC actor — device={DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    stats    = np.load(STATS_PATH, allow_pickle=False)
    obs_mean = np.array(stats["obs_mean"], dtype=np.float32)
    obs_std  = np.array(stats["obs_std"],  dtype=np.float32)

    config = make_config()
    uhrc   = load_bc(BC_CHECKPOINT, config)
    actor  = TD3Actor(uhrc).to(DEVICE)

    # Frozen BC anchor — never updated, used only to compute BC penalty
    frozen_bc = copy.deepcopy(actor).to(DEVICE)
    for p in frozen_bc.parameters(): p.requires_grad_(False)
    frozen_bc.eval()
    print("  ✅  Frozen BC anchor initialised")

    actor_target = copy.deepcopy(actor).to(DEVICE)
    for p in actor_target.parameters(): p.requires_grad_(False)

    critic        = TD3Critic().to(DEVICE)
    critic_target = copy.deepcopy(critic).to(DEVICE)
    for p in critic_target.parameters(): p.requires_grad_(False)

    print(f"  Actor  params : {sum(p.numel() for p in actor.parameters()):,}")
    print(f"  Critic params : {sum(p.numel() for p in critic.parameters()):,}")

    actor_opt  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    buffer = ReplayBuffer(BUFFER_SIZE, ForestEnv.obs_dim, ForestEnv.act_dim)
    env    = ForestEnv(obs_mean, obs_std)

    # Pre-compute normalised "at-goal" obs values for HER
    # raw=0 normalises to -mean/std (NOT 0.0 — that would be "at average dist")
    her_goal_rel_norm = np.array([
        -obs_mean[i] / max(float(obs_std[i]), 1e-3) for i in range(9, 13)
    ], dtype=np.float32)

    obs   = env.reset()
    carry: UHRCCarry | None = None
    ep_reward = 0.0

    total_steps        = 0
    update_count       = 0
    best_success       = 0.0
    ep_rewards:        list[float] = []
    ep_successes:      list[bool]  = []
    her_ep:            list[tuple] = []
    qmean_freeze_count = 0
    last_buf_size      = 0
    buf_frozen_count   = 0

    acc = {"critic_loss": 0., "actor_loss": 0., "q_target": 0.,
           "q_actor": 0., "bc_pen": 0., "n": 0}

    # Track Qmean history for dead critic detection — only warn if truly stuck
    qmean_history: list[float] = []

    print(f"\n{'Step':>9}  {'MeanRew':>9}  {'Succ%':>7}  "
          f"{'CritLoss':>9}  {'Qatr':>7}  {'Qact':>7}  {'BCpen':>7}  "
          f"{'Noise':>6}  {'Buf':>7}")
    print("─" * 88)

    while total_steps < TOTAL_TIMESTEPS:

        # ── Collect one step ──────────────────────────────────────────────────
        if total_steps < WARMUP_STEPS:
            action_np      = np.zeros(ForestEnv.act_dim, dtype=np.float32)
            action_np[0]   = np.random.uniform(9.3, 10.3)
            action_np[1:3] = np.random.uniform(-0.05, 0.05, 2)
            subgoal_np     = None
        else:
            obs_t              = torch.tensor(obs, dtype=torch.float32,
                                              device=DEVICE).unsqueeze(0)
            action_np, subgoal_np, carry = actor.collect_action(obs_t, carry)
            carry              = detach_carry(carry)
            sigma              = exploration_noise(total_steps)
            action_np          = (action_np +
                                  np.random.normal(0, sigma, size=4).astype(np.float32))

        action_np = np.clip(action_np,
                            [0., -0.5, -0.5, -0.5],
                            [20.,  0.5,  0.5,  0.5]).astype(np.float32)

        obs_next, reward, done, info = env.step(action_np, subgoal_np)

        # Divergence guard — NaN obs_next
        if not np.isfinite(np.asarray(obs_next, dtype=np.float32)).all():
            done     = True
            reward   = -50.
            obs_next = np.zeros(ForestEnv.obs_dim, dtype=np.float32)

        # Zero-obs guard — env stuck after previous divergence patch
        obs_arr = np.asarray(obs, dtype=np.float32)
        if not np.isfinite(obs_arr).all() or (np.abs(obs_arr).max() < 1e-6):
            obs    = env.reset()
            carry  = None
            her_ep = []
            total_steps += 1
            continue

        # terminal=True ONLY for crash/success — timeout bootstraps normally
        terminal = bool(info.get("crashed") or info.get("reached")) if \
                   isinstance(info, dict) else False
        buffer.add(obs, action_np, reward, obs_next, float(terminal))
        her_ep.append((obs.copy(), action_np.copy(), obs_next.copy()))
        ep_reward   += reward
        total_steps += 1

        if done:
            ep_rewards.append(ep_reward)
            ep_successes.append(bool(info["reached"]))

            # HER — relabel timeout final position as synthetic goal
            if info.get("timeout") and len(her_ep) > 0 and not info.get("crashed"):
                last_obs, last_act, last_next = her_ep[-1]
                her_obs      = last_next.copy()
                her_obs_next = last_next.copy()
                for idx in [9, 10, 11, 12]:
                    her_obs[idx]      = her_goal_rel_norm[idx - 9]
                    her_obs_next[idx] = her_obs[idx]
                buffer.add(her_obs, last_act, 100.0, her_obs_next, 1.0)

            her_ep    = []
            ep_reward = 0.
            obs       = env.reset()
            carry     = None
        else:
            obs = obs_next

        # ── Gradient updates ──────────────────────────────────────────────────
        if len(buffer) >= WARMUP_STEPS:
            if not actor_healthy(actor):
                reload_bc(actor, actor_target, config, actor_opt)

            for _ in range(UPDATES_PER_STEP):
                m = td3_update(actor, actor_target, critic, critic_target,
                               frozen_bc, actor_opt, critic_opt,
                               buffer, update_count)
                acc["critic_loss"] += m["critic_loss"]
                acc["actor_loss"]  += m["actor_loss"]
                acc["q_target"]    += m["q_target"]
                acc["q_actor"]     += m["q_actor"]
                acc["bc_pen"]      += m["bc_pen"]
                acc["n"]           += 1
                update_count       += 1

            # Qmean guard — temporary freeze when critic overestimates
            recent_succ  = float(np.mean(ep_successes[-20:])) if ep_successes else 0.
            recent_qmean = acc["q_target"] / max(acc["n"], 1)
            if recent_succ < 0.01 and recent_qmean > Q_TARGET_CLIP * 0.8:
                if qmean_freeze_count < QMEAN_FREEZE_LIMIT:
                    update_count += 1
                    qmean_freeze_count += 1
                    if total_steps % LOG_EVERY < UPDATES_PER_STEP:
                        print(f"  ⚠️  Qmean={recent_qmean:.2f} — actor frozen "
                              f"({qmean_freeze_count}/{QMEAN_FREEZE_LIMIT})")
                else:
                    print(f"  ❌  Freeze limit — reloading BC")
                    reload_bc(actor, actor_target, config, actor_opt)
                    qmean_freeze_count = 0
            else:
                qmean_freeze_count = 0

        # ── Logging ───────────────────────────────────────────────────────────
        if total_steps % LOG_EVERY == 0:
            mean_rew  = float(np.mean(ep_rewards[-20:]))   if ep_rewards   else 0.
            succ_rate = float(np.mean(ep_successes[-20:]))*100 if ep_successes else 0.
            n         = max(acc["n"], 1)
            buf_grew  = len(buffer) > last_buf_size
            qatr      = acc["q_target"]    / n
            qact      = acc["q_actor"]     / n
            bcpen     = acc["bc_pen"]      / n
            cl        = acc["critic_loss"] / n

            print(f"{total_steps:>9,}  {mean_rew:>9.2f}  {succ_rate:>6.1f}%  "
                  f"{cl:>9.4f}  {qatr:>7.3f}  {qact:>7.3f}  {bcpen:>7.4f}  "
                  f"{exploration_noise(total_steps):>6.3f}  {len(buffer):>7,}")

            # Dead critic detection — only warn if Qmean is truly STUCK (flat)
            # Qmean=-0.12 early in training is CORRECT, not a dead critic.
            # A dead critic has zero variance across intervals.
            qmean_history.append(qatr)
            if len(qmean_history) > 5:
                qmean_history.pop(0)
            if len(qmean_history) == 5:
                qmean_range = max(qmean_history) - min(qmean_history)
                if cl < 1e-4 and qmean_range < 0.005 and total_steps > WARMUP_STEPS * 5:
                    print(f"  ⚠️  Dead critic: Qmean={qatr:.3f} flat "
                          f"(range={qmean_range:.4f}) — reinitialising")
                    # Critic converged to mean return — cannot provide gradient direction.
                    # Reinitialise weights but keep buffer (transitions still valid).
                    critic        = TD3Critic().to(DEVICE)
                    critic_target = copy.deepcopy(critic).to(DEVICE)
                    for p in critic_target.parameters(): p.requires_grad_(False)
                    critic_opt    = optim.Adam(critic.parameters(), lr=LR_CRITIC)
                    qmean_history.clear()
                    print(f"  ✅  Critic reinitialised")

            # Buffer frozen monitor
            if not buf_grew:
                buf_frozen_count += 1
                print(f"  ⚠️  Buffer frozen ({buf_frozen_count}) — forcing env reset")
                obs = env.reset(); carry = None; her_ep = []
                if buf_frozen_count >= 3:
                    print(f"  ❌  Buffer frozen {buf_frozen_count} — reloading BC")
                    reload_bc(actor, actor_target, config, actor_opt)
                    buf_frozen_count = 0
            else:
                buf_frozen_count = 0

            last_buf_size = len(buffer)
            acc = {"critic_loss": 0., "actor_loss": 0., "q_target": 0.,
                   "q_actor": 0., "bc_pen": 0., "n": 0}

            torch.save({
                "total_steps": total_steps, "update_count": update_count,
                "actor":       actor.state_dict(),
                "critic":      critic.state_dict(),
                "actor_opt":   actor_opt.state_dict(),
                "critic_opt":  critic_opt.state_dict(),
                "best_success": best_success,
            }, os.path.join(SAVE_DIR, "td3_latest.pth"))

            if succ_rate > best_success and total_steps > WARMUP_STEPS:
                best_success       = succ_rate
                qmean_freeze_count = 0
                torch.save(actor.uhrc.state_dict(),
                           os.path.join(SAVE_DIR, "uhrc_td3_best.pth"))
                print(f"  ✅ New best: {best_success:.1f}% → uhrc_td3_best.pth")

        if DEVICE == "cuda" and total_steps % 20_000 == 0:
            torch.cuda.empty_cache(); gc.collect()

    print(f"\nDone. Best: {best_success:.1f}%")


if __name__ == "__main__":
    train()