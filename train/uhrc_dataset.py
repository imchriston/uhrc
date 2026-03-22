"""
ControlDataset
Loads the forest_data.npz produced by generate_forest_data.py and serves
windowed sequence samples for offline behavioural cloning.

Key design decisions
─────────────────────
• obs      (49-dim body-frame) — z-scored so the encoder sees unit-variance inputs
• actions  (4-dim wrench)      — NOT normalised; model predicts raw Newtons/Nm
• subgoals (3-dim velocity)    — NOT normalised; model predicts raw m/s
  (magnitude carries urgency information that z-scoring would destroy)
• Windows never cross episode boundaries (avoids leaking terminal states into
  the middle of sequences)
• Short windows at episode starts are left-padded with zeros
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class ControlDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 8, normalize: bool = True):
        data = np.load(data_path, allow_pickle=False)

        obs        = data["obs"].astype(np.float32)        # [N, 49]
        actions    = data["actions"].astype(np.float32)    # [N, 4]   raw Newtons/Nm
        subgoals   = data["subgoals"].astype(np.float32)   # [N, 3]   raw m/s
        episode_id = data["episode_id"].astype(np.int32)   # [N]

        # ── Normalisation stats ───────────────────────────────────────────────
        # Computed from full dataset so they're always consistent with the data.
        # Actions and subgoals intentionally excluded — raw physical units only.
        self.obs_mean = obs.mean(0).astype(np.float32)
        self.obs_std  = (obs.std(0) + 1e-6).astype(np.float32)

        # Still expose act/sub stats for diagnostics / controller loading
        self.act_mean = actions.mean(0).astype(np.float32)
        self.act_std  = (actions.std(0) + 1e-6).astype(np.float32)
        self.sub_mean = subgoals.mean(0).astype(np.float32)
        self.sub_std  = (subgoals.std(0) + 1e-6).astype(np.float32)

        if normalize:
            obs = (obs - self.obs_mean) / self.obs_std
            # actions and subgoals deliberately left in raw units

        self.obs      = torch.from_numpy(obs)
        self.actions  = torch.from_numpy(actions)    # raw Newtons/Nm
        self.subgoals = torch.from_numpy(subgoals)   # raw m/s

        # ── Build sequence windows respecting episode boundaries ──────────────
        self.seq_len              = seq_len
        self.windows: list[tuple[int, int]] = []
        _episode_id_per_window: list[int] = []

        unique_eps = np.unique(episode_id)
        for ep in unique_eps:
            idxs = np.where(episode_id == ep)[0]
            s, e = int(idxs[0]), int(idxs[-1]) + 1
            ep_int = int(ep)
            for i in range(s, e):
                win_start = max(s, i - seq_len + 1)
                self.windows.append((win_start, i + 1))
                _episode_id_per_window.append(ep_int)

        self.episode_id_per_window: np.ndarray = np.array(_episode_id_per_window, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        start, end = self.windows[idx]
        T          = self.seq_len

        obs_win = self.obs[start:end]       # [<=T, 49]
        act_win = self.actions[start:end]   # [<=T,  4]
        sub_win = self.subgoals[start:end]  # [<=T,  3]

        # Left-pad short windows at episode starts
        pad = T - obs_win.shape[0]
        if pad > 0:
            obs_win = torch.cat([torch.zeros(pad, obs_win.shape[-1]), obs_win], dim=0)
            act_win = torch.cat([torch.zeros(pad, act_win.shape[-1]), act_win], dim=0)
            sub_win = torch.cat([torch.zeros(pad, sub_win.shape[-1]), sub_win], dim=0)

        return {
            "state":   obs_win,   # [seq_len, 49]  z-scored
            "action":  act_win,   # [seq_len,  4]  raw Newtons/Nm
            "subgoal": sub_win,   # [seq_len,  3]  raw m/s
        }