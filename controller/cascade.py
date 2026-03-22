import numpy as np
from numpy.typing import NDArray
from typing import Optional
import dynamics
import utils.quat_euler as quat_euler
from controller.position import PositionPI
from controller.attitude import AttitudePID

class CascadedPosAttController:
    """
    Outer-loop position PI (x,y) + inner-loop AttitudePID (φ,θ,ψ,z).
    Includes Safety Clamping to prevent flip-overs on large steps.
    """

    def __init__(
        self,
        dyn: dynamics.QuadrotorDynamics,
        pos_pi: PositionPI,
        att_pid: AttitudePID,
        sample_time: float = 0.01,
        max_tilt_deg: float = 35.0  # <--- NEW: Max tilt safety limit
    ):
        self.dyn = dyn
        self.pos_pi = pos_pi
        self.att_pid = att_pid
        self.sample_time = float(sample_time)
        self.max_tilt = np.deg2rad(max_tilt_deg) # Convert to radians

        # position & yaw references
        self.r_ref = np.zeros(3)   # [x, y, z] in ENU
        self.psi_ref = 0.0

        self._last_t: Optional[float] = None

        # logging buffers
        self.log_t = []          
        self.log_r_ref = []      
        self.log_phi_d = []      
        self.log_theta_d = []    
        self.log_psi_ref = []    

    def set_position_ref(self, x: float, y: float, z: float):
        self.r_ref[:] = [x, y, z]

    def set_yaw_ref(self, psi: float):
        self.psi_ref = float(psi)

    def reset(self, x0: NDArray[np.float64]):
        r_I, v_I, q_BI, omega_B, _ = self.dyn.unpack_state(x0)
        self.pos_pi.reset(r_I, self.r_ref)
        self.att_pid.reset()
        self._last_t = None

    def __call__(self, t: float, x: NDArray[np.float64]):
        dt = self.sample_time if self._last_t is None else max(1e-4, min(0.1, float(t - self._last_t)))
        self._last_t = t

        # unpack current state
        r_I, v_I, q_BI, omega_B, _ = self.dyn.unpack_state(x)
        phi, theta, psi = quat_euler.euler_from_q(q_BI)

        # ---- OUTER LOOP ----
        # 1. Get desired accelerations (unbounded)
        a_des = self.pos_pi.step(self.r_ref, r_I, v_I, dt)
        ax_des, ay_des, _ = a_des 

        # ---- MAPPING & CLAMPING ----
        g = self.dyn.p.g
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # 2. Linear Map acceleration -> angle
        # (This is valid only for small angles, so we MUST clamp it)
        theta_d_raw = (ax_des * cpsi + ay_des * spsi) / g
        phi_d_raw   = (ax_des * spsi - ay_des * cpsi) / g

        # 3. SAFETY CLAMP (The Fix for Overshoot)
        # Prevents requesting 700 degree tilts on 40m errors
        theta_d = np.clip(theta_d_raw, -self.max_tilt, self.max_tilt)
        phi_d   = np.clip(phi_d_raw,   -self.max_tilt, self.max_tilt)

        # ---- INNER LOOP ----
        refs_att = {
            'phi':   float(phi_d),
            'theta': float(theta_d),
            'psi':   float(self.psi_ref),
            'z':     float(self.r_ref[2]),
        }

        u = self.att_pid.step(x, refs_att, dt)
        
        # log references
        self.log_t.append(t)
        self.log_r_ref.append(self.r_ref.copy())
        self.log_phi_d.append(phi_d)
        self.log_theta_d.append(theta_d)
        self.log_psi_ref.append(self.psi_ref)
        
        return "body_wrench", u