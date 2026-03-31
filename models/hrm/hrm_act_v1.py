import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel




def uhrc_trunc_normal_(tensor: torch.Tensor, std: float = 1.0):
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, std=std, a=-2*std, b=2*std)
    return tensor


def uhrc_rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x.to(dtype)


class UHRC_Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            uhrc_trunc_normal_(torch.empty(out_features, in_features),
                               std=1.0 / math.sqrt(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class UHRC_SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        inter = int(round(expansion * hidden_size * 2 / 3))
        inter = ((inter + 255) // 256) * 256          # round up to multiple of 256
        self.gate_up = UHRC_Linear(hidden_size, inter * 2, bias=False)
        self.down    = UHRC_Linear(inter, hidden_size,     bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class UHRC_Attention(nn.Module):
    """Multi-head self-attention without flash-attn dependency."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.qkv = UHRC_Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = UHRC_Linear(hidden_size, hidden_size,     bias=False)

    def forward(self, x: torch.Tensor,
                cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)   # each [B, T, H, D]

        if cos is not None and sin is not None:
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        # [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        return self.out(attn)

    @staticmethod
    def _apply_rope(x: torch.Tensor,
                    cos: torch.Tensor,
                    sin: torch.Tensor) -> torch.Tensor:
        """
        Apply Rotary Position Embedding to q or k.

        Args:
            x   : [B, T, num_heads, head_dim]      — query or key tensor
            cos : [max_seq_len, head_dim // 2]      — from UHRC_RoPE
            sin : [max_seq_len, head_dim // 2]

        Returns:
            [B, T, num_heads, head_dim]  with RoPE applied.
        """
        T = x.shape[1]

        # Slice to actual sequence length, add broadcast dims for B and num_heads
        cos = cos[:T].unsqueeze(0).unsqueeze(2)   # [1, T, 1, head_dim // 2]
        sin = sin[:T].unsqueeze(0).unsqueeze(2)   # [1, T, 1, head_dim // 2]

        # Split into even / odd frequency pairs — each [B, T, num_heads, head_dim//2]
        x1 = x[..., ::2]    # even indices
        x2 = x[..., 1::2]   # odd  indices

        # Standard RoPE rotation:
        #   [x1, x2] → [x1·cos − x2·sin,  x2·cos + x1·sin]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        # Interleave back to original layout: [B, T, H, head_dim]
        # torch.stack on last dim then flatten restores even/odd interleaving.
        return torch.stack([out1, out2], dim=-1).flatten(-2)



class UHRC_RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t   = torch.arange(max_seq_len).float()
        emb = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached   # [T, D//2]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class UHRC_Config(BaseModel):
    # Observation layout: v_B(3)|w_B(3)|g_B(3)|goal_rel_B(3)|goal_dist(1)|Omega(4)|lidar(32)
    state_dim:           int   = 45
    lidar_dim:           int   = 32
    lidar_conv_channels: int   = 16

    action_dim:  int   = 4    # [Fz, τx, τy, τz]
    subgoal_dim: int   = 3    # body-frame velocity [vx, vy, vz] m/s

    # Carry / temporal memory
    carry_len:   int   = 16   # number of past hidden states kept in carry
                               # (replaces seq_len window — true rolling buffer)

    hidden_size: int   = 256
    expansion:   float = 4.0
    num_heads:   int   = 4
    rms_norm_eps: float = 1e-5
    rope_theta:  float = 10000.0

    hover_thrust: float = 9.81

    # FIX 1: H_cycles must be >= 2 for mutual refinement.
    # With H_cycles=1, L ran using old z_H, H updated, done — no feedback.
    # With H_cycles=2, round-2 L sees the H that already saw round-1 L.
    H_cycles: int = 2     # was 1 — CHANGED
    L_cycles: int = 2

    H_layers: int = 2
    L_layers: int = 2

    detach_carry: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Carry dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UHRCCarry:
    z_H: torch.Tensor   # [B, carry_len, hidden_size]
    z_L: torch.Tensor   # [B, carry_len, hidden_size]


# ─────────────────────────────────────────────────────────────────────────────
# Transformer block (shared by H and L)
# ─────────────────────────────────────────────────────────────────────────────

class UHRCBlock(nn.Module):
    def __init__(self, config: UHRC_Config):
        super().__init__()
        self.attn     = UHRC_Attention(config.hidden_size, config.num_heads)
        self.mlp      = UHRC_SwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, x: torch.Tensor,
                cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = uhrc_rms_norm(x + self.attn(x, cos, sin), self.norm_eps)
        x = uhrc_rms_norm(x + self.mlp(x),             self.norm_eps)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning module
# ─────────────────────────────────────────────────────────────────────────────

class UHRCReasoningModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, z: torch.Tensor, injection: torch.Tensor,
                cos=None, sin=None) -> torch.Tensor:
        """z += injection ONCE, then run attention layers."""
        z = z + injection
        for layer in self.layers:
            z = layer(z, cos, sin)
        return z


# ─────────────────────────────────────────────────────────────────────────────
# UHRC Inner
# ─────────────────────────────────────────────────────────────────────────────

class UHRC_Inner(nn.Module):
    def __init__(self, config: UHRC_Config):
        super().__init__()
        self.config = config
        H = config.hidden_size

        # ── Input encoder ────────────────────────────────────────────────────
        scalar_dim    = config.state_dim - config.lidar_dim   # 17
        lidar_out_dim = config.lidar_conv_channels * config.lidar_dim

        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, config.lidar_conv_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(config.lidar_conv_channels, config.lidar_conv_channels,
                      kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(start_dim=1),
        )
        embed_scale        = math.sqrt(H)
        self.embed_scale   = embed_scale
        self.scalar_enc_1  = UHRC_Linear(scalar_dim,  H // 2)
        self.scalar_enc_2  = UHRC_Linear(H // 2,      H // 2)
        self.lidar_proj    = UHRC_Linear(lidar_out_dim, H // 2)
        self.fusion        = UHRC_Linear(H,             H)

        # ── H / L reasoning modules ──────────────────────────────────────────
        self.H_level = UHRCReasoningModule(
            [UHRCBlock(config) for _ in range(config.H_layers)])
        self.L_level = UHRCReasoningModule(
            [UHRCBlock(config) for _ in range(config.L_layers)])


        self.planning_head = nn.Sequential(
            UHRC_Linear(H, 64),
            nn.SiLU(),
            UHRC_Linear(64, config.subgoal_dim),
        )

        # ── Subgoal → L gated injection ───────────────────────────────────────
        # attention layers process the subgoal signal
        self.subgoal_proj = nn.Sequential(
            UHRC_Linear(config.subgoal_dim, H),
            nn.SiLU(),
        )
        self.subgoal_gate = nn.Sequential(
            UHRC_Linear(config.subgoal_dim, H),
            nn.Sigmoid(),
        )

        # ── Action head (L → wrench) ──────────────────────────────────────────
        self.action_head = UHRC_Linear(H, config.action_dim)

        # ── Positional encoding ───────────────────────────────────────────────
        # carry_len+1 to accommodate the new observation appended each step.
        self.rotary_emb = UHRC_RoPE(
            dim=H // config.num_heads,
            max_seq_len=config.carry_len + 1,
            base=config.rope_theta,
        )

        # ── Carry initialisations ─────────────────────────────────────────────
        self.H_init = nn.Buffer(
            uhrc_trunc_normal_(torch.empty(H), std=1.0), persistent=True)
        self.L_init = nn.Buffer(
            uhrc_trunc_normal_(torch.empty(H), std=1.0), persistent=True)

        # ── Head initialisations ──────────────────────────────────────────────
        with torch.no_grad():
            # action_head: hover thrust bias on Fz
            assert isinstance(self.action_head, UHRC_Linear)
            nn.init.normal_(self.action_head.weight, std=0.01)
            assert self.action_head.bias is not None
            self.action_head.bias.zero_()
            self.action_head.bias[0] = config.hover_thrust

            # planning_head final layer: start near zero velocity
            planning_out = self.planning_head[-1]
            assert isinstance(planning_out, UHRC_Linear)
            nn.init.normal_(planning_out.weight, std=0.01)
            assert planning_out.bias is not None
            planning_out.bias.zero_()

            # Gate starts mostly open (σ(+2) ≈ 0.88)
            gate_linear = self.subgoal_gate[0]
            assert isinstance(gate_linear, UHRC_Linear)
            assert gate_linear.bias is not None
            gate_linear.bias.fill_(2.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_obs(self, state: torch.Tensor) -> torch.Tensor:
        """[B, 49] → [B, H]"""
        scalar = state[..., : self.config.state_dim - self.config.lidar_dim]
        lidar  = state[..., self.config.state_dim - self.config.lidar_dim :]

        s = self.scalar_enc_2(torch.tanh(
            self.embed_scale * self.scalar_enc_1(scalar)))
        l = self.lidar_proj(
            self.lidar_encoder(lidar.unsqueeze(1)))    # [B, 1, 32] → conv
        return self.fusion(torch.cat([s, l], dim=-1))  # [B, H]

    def empty_carry(self, batch_size: int, device, dtype) -> UHRCCarry:
        H, CL = self.config.hidden_size, self.config.carry_len
        return UHRCCarry(
            z_H=self.H_init.to(device, dtype).view(1, 1, H)
                           .expand(batch_size, CL, H).clone(),
            z_L=self.L_init.to(device, dtype).view(1, 1, H)
                           .expand(batch_size, CL, H).clone(),
        )

    def _roll_carry(self, carry_vec: torch.Tensor,
                    new_token: torch.Tensor) -> torch.Tensor:
        """
        FIX 4: Rolling carry buffer.
        Discard oldest position, append new_token at the end.
        carry_vec : [B, carry_len,   H]
        new_token : [B, H]
        → [B, carry_len, H]  (oldest dropped, newest appended)
        """
        return torch.cat([carry_vec[:, 1:, :],
                          new_token.unsqueeze(1)], dim=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        carry: Optional[UHRCCarry],
        state: torch.Tensor,          # [B, 49]
    ) -> Tuple[UHRCCarry, torch.Tensor, torch.Tensor]:

        B      = state.shape[0]
        device = state.device
        dtype  = state.dtype

        if carry is None or carry.z_H.shape[0] != B:
            carry = self.empty_carry(B, device, dtype)

        cos, sin = self.rotary_emb()

        # Encode current observation → [B, H]
        new_emb = self._encode_obs(state)

        # seq_H/seq_L : [B, carry_len+1, H]
        # Position 0 = oldest, position carry_len = current step.
        seq_H = torch.cat([carry.z_H, new_emb.unsqueeze(1)], dim=1)
        seq_L = torch.cat([carry.z_L, new_emb.unsqueeze(1)], dim=1)

        z_H = seq_H
        z_L = seq_L

        # Cycle structure (same as HRM):
        #   Round 1: L refines using old z_H  →  H refines using new z_L
        #   Round 2: L refines using UPDATED z_H  →  H refines again
        # After round 2, z_L has seen H's updated plan, z_H has seen L's
        # updated execution state.  Neither happened with H_cycles=1.
        for _ in range(self.config.H_cycles):
            for _ in range(self.config.L_cycles):
                # L gets raw sensor context + H's current abstract plan
                z_L = self.L_level(z_L, z_H, cos=cos, sin=sin)
            # H gets only L's state (forces H to abstract, not re-read sensors)
            z_H = self.H_level(z_H, z_L, cos=cos, sin=sin)

        # ── H → subgoal ───────────────────────────────────────────────────────
        # Use the most recent position (current timestep) to read the subgoal.
        subgoal = self.planning_head(z_H[:, -1, :])   # [B, 3]

        # broadcast the projected subgoal across ALL positions in z_L
        # so the final L attention pass can integrate it with full context.
        gate            = self.subgoal_gate(subgoal)          
        subgoal_context = self.subgoal_proj(subgoal) * gate   
        # Add to every position — L can attend across all of them
        z_L_final = z_L + subgoal_context.unsqueeze(1)        # [B, T, H]
        # One final L pass so attention layers process the subgoal signal
        for layer in self.L_level.layers:
            z_L_final = layer(z_L_final, cos, sin)

        # ── L → action ────────────────────────────────────────────────────────
        action = self.action_head(z_L_final[:, -1, :])        # [B, 4]

        # ── Carry update ──────────────────────────────────────────────────────
        # position i in next carry = what z_H/z_L looked like i steps ago.
        new_z_H = self._roll_carry(carry.z_H, z_H[:, -1, :].detach()
                                   if self.config.detach_carry
                                   else z_H[:, -1, :])
        new_z_L = self._roll_carry(carry.z_L, z_L_final[:, -1, :].detach()
                                   if self.config.detach_carry
                                   else z_L_final[:, -1, :])

        new_carry = UHRCCarry(z_H=new_z_H, z_L=new_z_L)
        return new_carry, action, subgoal


# ─────────────────────────────────────────────────────────────────────────────
# Outer wrapper
# ─────────────────────────────────────────────────────────────────────────────

class UHRC(nn.Module):
    def __init__(self, config: UHRC_Config):
        super().__init__()
        self.config = config
        self.inner  = UHRC_Inner(config)

    def forward(
        self,
        state: torch.Tensor,
        carry: Optional[UHRCCarry] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, UHRCCarry]:
        """
        Args:
            state  : [B, 49]
            carry  : UHRCCarry or None (auto-init)
        Returns:
            action  : [B, 4]   motor wrench [Fz, τx, τy, τz]
            subgoal : [B, 3]   body-frame velocity setpoint m/s
            carry   : updated UHRCCarry
        """
        if state.ndim == 3:
            # BC training passes [B, T, 45] 
            # so the carry rolls correctly through time.
            actions:  list[torch.Tensor] = []
            subgoals: list[torch.Tensor] = []
            rolling_carry: UHRCCarry = (
                carry if carry is not None
                else self.inner.empty_carry(state.shape[0], state.device, state.dtype)
            )
            for t in range(state.shape[1]):
                rolling_carry, act, sub = self.inner(rolling_carry, state[:, t, :])
                actions.append(act)
                subgoals.append(sub)
            return torch.stack(actions, dim=1), torch.stack(subgoals, dim=1), rolling_carry

        # Inference: single step [B, 45]
        new_carry, action, subgoal = self.inner(carry, state)
        return action, subgoal, new_carry