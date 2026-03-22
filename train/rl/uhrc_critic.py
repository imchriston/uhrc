
from __future__ import annotations
import math
import torch
import torch.nn as nn
from models.layers import CastedLinear


class UHRCCritic(nn.Module):
    """
    State-value function V(s) for PPO.

    Input:  [B, 49]  or  [B, T, 49]  normalised observation
    Output: [B, 1]   scalar value estimate
    """

    def __init__(self, state_dim: int = 49, lidar_dim: int = 32,
                 hidden_size: int = 256, lidar_conv_ch: int = 16) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.lidar_dim  = lidar_dim
        self.embed_scale = math.sqrt(hidden_size)

        scalar_dim    = state_dim - lidar_dim          # 17
        lidar_out_dim = lidar_conv_ch * lidar_dim      # 16 × 32 = 512

        # ── Lidar conv encoder (mirrors actor) ────────────────────────────────
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, lidar_conv_ch, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(lidar_conv_ch, lidar_conv_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(start_dim=1),
        )

        # ── Scalar encoder (mirrors actor) ────────────────────────────────────
        self.scalar_enc_1 = CastedLinear(scalar_dim,      hidden_size // 2, bias=True)
        self.scalar_enc_2 = CastedLinear(hidden_size // 2, hidden_size // 2, bias=True)
        self.lidar_proj   = CastedLinear(lidar_out_dim,   hidden_size // 2, bias=True)
        self.fusion       = CastedLinear(hidden_size,     hidden_size,      bias=True)

        # ── Value head ────────────────────────────────────────────────────────
        self.value_head = nn.Sequential(
            CastedLinear(hidden_size, 128, bias=True),
            nn.SiLU(),
            CastedLinear(128, 1, bias=True),
        )

        # Initialise value head near zero so critic starts with V≈0
        with torch.no_grad():
            nn.init.normal_(self.value_head[-1].weight, std=0.01)   # type: ignore[arg-type]
            if self.value_head[-1].bias is not None:                 # type: ignore[union-attr]
                self.value_head[-1].bias.zero_()                     # type: ignore[union-attr]

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode [B, 49] → [B, hidden_size]."""
        scalar = state[..., : self.state_dim - self.lidar_dim]
        lidar  = state[..., self.state_dim  - self.lidar_dim :]

        B = state.shape[0]

        s_raw = self.embed_scale * self.scalar_enc_1(scalar)
        s_emb = self.scalar_enc_2(torch.tanh(s_raw))

        l_raw = lidar.reshape(B, 1, self.lidar_dim)
        l_emb = self.lidar_proj(self.lidar_encoder(l_raw))

        return self.fusion(torch.cat([s_emb, l_emb], dim=-1))   # [B, H]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, 49] or [B, T, 49] — normalised observation
                   If [B, T, 49], the LAST timestep is used for the value
                   estimate (consistent with how the actor produces actions).
        Returns:
            value: [B, 1]
        """
        if state.ndim == 3:
            state = state[:, -1, :]   # use last step: [B, 49]
        return self.value_head(self._encode(state))