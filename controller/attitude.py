from dataclasses import dataclass
import math
import numpy as np
import dynamics
from numpy.typing import NDArray
from utils import quat_euler as quat
import controller.pid as pid

def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

class AttitudePID:
    """
    Composition-based Attitude + Altitude controller.
    - Input: References (phi, theta, psi, z)
    - Output: Body Wrench (Thrust, tau_x, tau_y, tau_z)
    """

    def __init__(
        self,
        params: dynamics.QuadrotorParams,
        gains_roll: pid.PIDGains,
        gains_pitch: pid.PIDGains,
        gains_yaw: pid.PIDGains,
        gains_z: pid.PIDGains,
        torque_limits=(0.6, 0.6, 0.35),
        thrust_limits=(0.0, None),
        az_limit: float = 6.0,
        d_cut_angles_hz: float = 10.0,
        d_cut_vz_hz: float = 5.0,
        sample_time: float = 0.01,
        yaw_dir: float = -1.0
    ):
        self.p = params
        self.yaw_sign = yaw_dir
        
        # Initialize Logs
        self.log_t = []       
        self.log_r_ref = []   
        self.log_phi_d = []   
        self.log_theta_d = [] 
        self.log_psi_ref = [] 

        # Limits
        self.tau_max = np.array(torque_limits, dtype=float)
        T_phys_max = 4.0 * self.p.kT * (self.p.max_omega ** 2)
        tmin, tmax = thrust_limits
        self.T_min = float(0.0 if tmin is None else tmin)
        self.T_max = float(T_phys_max if tmax is None else tmax)

        # PIDs
        self.pid_roll = pid.PID(gains_roll, -self.tau_max[0], self.tau_max[0], -self.tau_max[0], self.tau_max[0], d_cutoff_hz=d_cut_angles_hz)
        self.pid_pitch = pid.PID(gains_pitch, -self.tau_max[1], self.tau_max[1], -self.tau_max[1], self.tau_max[1], d_cutoff_hz=d_cut_angles_hz)
        self.pid_yaw = pid.PID(gains_yaw, -self.tau_max[2], self.tau_max[2], -self.tau_max[2], self.tau_max[2], d_cutoff_hz=d_cut_angles_hz)
        self.pid_z = pid.PID(gains_z, -az_limit, az_limit, -az_limit, az_limit, d_cutoff_hz=d_cut_vz_hz)

        # State
        self.refs = {'phi': 0.0, 'theta': 0.0, 'psi': 0.0, 'z': 0.0}
        self.sample_time = float(sample_time)
        self._last_t = None

    def set_refs(self, *, phi=None, theta=None, psi=None, z=None):
        if phi   is not None: self.refs['phi']   = float(phi)
        if theta is not None: self.refs['theta'] = float(theta)
        if psi   is not None: self.refs['psi']   = float(psi)
        if z     is not None: self.refs['z']     = float(z)

    def reset(self):
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()
        self.pid_z.reset()
        self._last_t = None

    def step(self, x: NDArray[np.float64], refs: dict, dt: float) -> np.ndarray:
        # 1. Unpack State
        r_I, v_I, q_BI, omega_B, _ = dynamics.QuadrotorDynamics.unpack_state(x)
        p, q, r = omega_B
        phi, theta, psi = quat.euler_from_q(q_BI)

        # 2. Unpack References
        phi_d   = float(refs.get('phi',   0.0))
        theta_d = float(refs.get('theta', 0.0))
        psi_d   = float(refs.get('psi',   0.0))
        z_d     = float(refs.get('z',     r_I[2]))

        # 3. Calc Errors
        e_phi   = _wrap_pi(phi_d   - phi)
        e_theta = _wrap_pi(theta_d - theta)
        e_psi   = _wrap_pi(psi_d   - psi)
        e_z     = z_d - r_I[2]

        # 4. PID Steps
        tau_x = self.pid_roll.step(e_phi,   dt, d_meas=p)
        tau_y = self.pid_pitch.step(e_theta, dt, d_meas=q)
        tau_z = self.pid_yaw.step(e_psi, dt, d_meas=r)
        a_z_cmd = self.pid_z.step(e_z, dt, d_meas=v_I[2])

        # 5. Thrust Mixing
        c_tilt = max(0.2, math.cos(phi) * math.cos(theta))
        T = self.p.mass * (self.p.g + a_z_cmd) / c_tilt
        T = float(np.clip(T, self.T_min, self.T_max))

        return np.array([T, tau_x, tau_y, tau_z], dtype=float)

    # --- THE FIXED CALL METHOD ---
    def __call__(self, t: float, x: NDArray[np.float64]):
        """
        Standard interface for the simulation loop.
        """
        # 1. Time Management
        dt = self.sample_time if self._last_t is None else max(1e-4, min(0.1, float(t - self._last_t)))
        self._last_t = t

        # 2. Run Control Logic (Using stored refs)
        u = self.step(x, self.refs, dt)

        # 3. Logging (Grab the current refs directly)
        self.log_t.append(t)
        # Log 0 for x/y ref since this is attitude mode, but keep z
        self.log_r_ref.append([0.0, 0.0, self.refs['z']]) 
        self.log_phi_d.append(self.refs['phi'])
        self.log_theta_d.append(self.refs['theta'])
        self.log_psi_ref.append(self.refs['psi'])

        return "body_wrench", u