from __future__ import annotations
import torch
import numpy as np
import os
from typing import Optional, Tuple, Literal
from numpy.typing import NDArray
from models.hrm.hrm_act_v1 import UHRC, UHRC_Config


def yaw_from_q_BI_wxyz_np(q_BI: NDArray[np.float64]) -> float:
    """
    Extract yaw from quaternion stored as q_BI = conj(q_IB), order [w,x,y,z].
    This returns the same psi used in q_from_euler(phi, theta, psi).
    """
    qw, qx, qy, qz = q_BI
    siny = 2.0 * (qx * qy - qw * qz)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny, cosy))


def wrap_to_pi_np(a: float) -> float:
    # stable wrap using atan2(sin, cos)
    return float(np.arctan2(np.sin(a), np.cos(a)))


class UHRCController:
    def __init__(self, model_path: str, stats_path: str, device: str = "cpu"):
        self.device = device

        # 1) Load Normalization Stats
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found at {stats_path}")

        stats = torch.load(stats_path, map_location=device)
        self.input_mean = stats["input_mean"].to(device)
        self.input_std = stats["input_std"].to(device)
        self.action_mean = stats["action_mean"].to(device)
        self.action_std = stats["action_std"].to(device)

        # Optional: stored by the dataset (recommended)
        self.input_mode = stats.get("input_mode", "state_ref")

        # Robust: match the model input dim to your saved stats
        state_dim = int(stats["input_mean"].numel())

        # 2) Initialize Model Architecture
        self.config = UHRC_Config(
            state_dim=state_dim,
            action_dim=4,
            hidden_size=256,
            H_layers=2,
            L_layers=2,
            H_cycles=1,
            L_cycles=2,
            seq_len=1,
            batch_size=512
        )
        self.model = UHRC(self.config).to(device)

        # 3) Load Weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        print(f"🧠 Loading UHRC (Tracking Mode) from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 4) Internal State
        self.carry = None
        self.ref = np.zeros(4, dtype=np.float64)

    def reset(self):
        self.carry = None

    def set_position_ref(self, x: float, y: float, z: float):
        self.ref[0] = x
        self.ref[1] = y
        self.ref[2] = z

    def set_yaw_ref(self, psi: float):
        self.ref[3] = psi

    def __call__(
        self,
        t: float,
        x: NDArray[np.float64],
        ref: Optional[NDArray[np.float64]] = None
    ) -> Tuple[Literal["body_wrench"], NDArray[np.float64]]:
        """
        Standard controller interface.
        """
        # 1) Handle Reference
        current_ref: NDArray[np.float64] = ref if ref is not None else self.ref
        expected_dim = int(self.input_mean.numel())

        if self.input_mode == "state_error":
            r = x[0:3]
            q_BI = x[6:10]
            psi = yaw_from_q_BI_wxyz_np(q_BI)

            e_pos = current_ref[0:3] - r
            e_yaw = wrap_to_pi_np(current_ref[3] - psi)
            yaw_feat = np.array([np.sin(e_yaw), np.cos(e_yaw)], dtype=np.float64)

            x_in = np.concatenate([x, e_pos, yaw_feat], axis=0).astype(np.float32)  # 22 dims

            # Build either 21D (e_yaw) or 22D (sin/cos) depending on stats
            if expected_dim == 21:
                tail = np.concatenate([e_pos, np.array([e_yaw], dtype=np.float64)], axis=0)
            elif expected_dim == 22:
                tail = np.concatenate([e_pos, np.array([np.sin(e_yaw), np.cos(e_yaw)], dtype=np.float64)], axis=0)
            else:
                raise RuntimeError(f"Unexpected expected_dim={expected_dim}. "
                                f"Stats imply {expected_dim}, but controller only supports 21 or 22 for state_error.")

            x_in = np.concatenate([x, tail], axis=0).astype(np.float32)

        else:
            # state_ref path should match stats too
            x_in = np.concatenate([x, current_ref], axis=0).astype(np.float32)

        # Safety check before torch conversion
        if x_in.shape[0] != expected_dim:
            raise RuntimeError(f"Input dim mismatch: built {x_in.shape[0]} but stats expect {expected_dim}. "
                            f"input_mode={self.input_mode}")


        x_tensor = torch.from_numpy(x_in).to(self.device)

        # 3) Normalize
        x_norm = (x_tensor - self.input_mean) / self.input_std

        # 4) Forward pass (recurrent)
        with torch.no_grad():
            u_norm, new_carry = self.model(x_norm.unsqueeze(0), carry=self.carry)
            self.carry = None

        # 5) Denormalize + postprocess
        u_tensor = u_norm[0] * self.action_std + self.action_mean
        u_out = u_tensor.cpu().numpy().astype(np.float64)
        # in __init__
        self.debug = True
        self.debug_log = []

        # inside __call__ after computing u_out
        if self.debug:
            # compute psi/e_yaw same way you build features
            q_BI = x[6:10]
            psi = yaw_from_q_BI_wxyz_np(q_BI)
            e_yaw = wrap_to_pi_np(current_ref[3] - psi)
            wz = float(x[12])

            self.debug_log.append({
                "t": float(t),
                "psi": psi,
                "psi_ref": float(current_ref[3]),
                "e_yaw": e_yaw,
                "wz": wz,
                "Tz": float(u_out[3]),
            })

        # Clips
        u_out[0] = np.clip(u_out[0], 0.0, 25.0)
        u_out[1] = np.clip(u_out[1], -1.0, 1.0)
        u_out[2] = np.clip(u_out[2], -1.0, 1.0)
        u_out[3] = np.clip(u_out[3], -0.5, 0.5)


        return "body_wrench", u_out