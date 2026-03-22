from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Tuple, Literal, Optional, Dict
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear
import math
from utils import quat_euler as quat_euler

# ----------------------------- Parameters -----------------------------

class QuadrotorParams:
    """
    Physical & model parameters for quadrotor 
    """
    mass: float = 1.0
    J: NDArray[np.float64] = np.diag([0.008, 0.008, 0.014])  # inertia matrix
    arm_length: float = 0.17
    kT: float = 1.9e-6               # thrust coeff: N / (rad/s)^2
    kM: float = 2.6e-8               # yaw moment coeff: N·m / (rad/s)^2
    max_rpm: float = 18000.0         # physical motor limits
    tau_motor: float = 0.035         # motor time constant (kept; unused in body_wrench path)
    cd_lin: NDArray[np.float64] = np.array([0.1, 0.1, 0.18])   # N/(m/s) linear drag diag
    cd_rot: NDArray[np.float64] = np.array([0.02, 0.02, 0.02]) # N·m/(rad/s) damping diag
    g: float = 9.81
    # rotor spin directions (+1 = CCW, -1 = CW)
    spin_dir: NDArray[np.float64] = np.array([-1.0, +1.0, -1.0, +1.0])
    # ENU frame
    frame: Literal["ENU"] = "ENU"

    def __post_init__(self):
        self.J = np.array(self.J, dtype=float)
        self.cd_lin = np.asarray(self.cd_lin, dtype=float)
        self.cd_rot = np.asarray(self.cd_rot, dtype=float)
        self.spin_dir = np.asarray(self.spin_dir, dtype=float)

    @property
    def max_omega(self) -> float:
        """Maximum rotor speed [rad/s]."""
        return (self.max_rpm * 2.0 * math.pi) / 60.0


def allocation_matrix(params: QuadrotorParams):
    """
    Mixer mapping y = [T, τx, τy, τz]^T = A @ s, with s = omega^2
    Rotor order: [FR(+l,+l), FL(+l,-l), RL(-l,-l), RR(-l,+l)]
    spin_dir = [-1, +1, -1, +1].
    """
    kf, km, l = params.kT, params.kM, params.arm_length
    sd = params.spin_dir
    A = np.array([
        [ kf,       kf,       kf,       kf      ],   # total thrust
        [ 0,    -l*kf,    0,     l*kf    ],   # τx
        [-l*kf,    0,     l*kf,     0    ],   # τy
        [ km*sd[0], km*sd[1], km*sd[2], km*sd[3] ]   # τz
    ], dtype=float)
    return A


# ----------------------------- Dynamics -----------------------------

class QuadrotorDynamics:
    """
    6-DOF quadrotor dynamics with quaternion attitude (ENU frame).
    State x = [ r_I(3), v_I(3), q_BI(4), ω_B(3), Ω(4) ].
    Inputs: u = [T, τx, τy, τz] 
    """

    def __init__(self, params: QuadrotorParams):
        self.p = params
        self.J = self.p.J
        self.J_inv = np.linalg.inv(self.J)
        # Rotor arm positions in body frame (for r×F torque check)
        l = self.p.arm_length
        self.r_i_B = np.array([
            [ +l, +l, 0.0],   # FR
            [ +l, -l, 0.0],   # FL 
            [ -l, -l, 0.0],   # RL
            [ -l, +l, 0.0],   # RR
        ], dtype=float)
        # mixer
        self.A = allocation_matrix(self.p)

    @staticmethod
    def state_size() -> int:
        return 3 + 3 + 4 + 3 + 4

    @staticmethod
    def unpack_state(x: NDArray[np.float64]):
        r_I = x[0:3]
        v_I = x[3:6]
        q_BI = quat_euler.q_normalize(x[6:10])
        omega_B = x[10:13]
        Omega = x[13:17]
        return r_I, v_I, q_BI, omega_B, Omega

    @staticmethod
    def pack_state(r_I, v_I, q_BI, omega_B, Omega):
        return np.concatenate([r_I, v_I, quat_euler.q_normalize(q_BI), omega_B, Omega])

    # ----------------- Input handling (BODY WRENCH ONLY) -----------------
    
    def _inputs_to_forces(self,
    
                          
                          u: NDArray[np.float64],
                          x: NDArray[np.float64],
                          input_type: Literal["body_wrench"]                          ):
        """
        Desired body wrench y = [T, τx, τy, τz] is mixed to squared speeds s = Ω^2 using:
             minimize ||A s - y||  subject to  0 ≤ s_i ≤ Ω_max^2
        Then T_i = kT s_i, Mz_i = kM s_i · spin_dir_i, and Ω_cmd = sqrt(s).
        """
        y = np.asarray(u, dtype=float).reshape(4,)
        A = self.A
        lb = np.zeros(4)
        ub = np.full(4, self.p.max_omega**2)

        sol = lsq_linear(A, y, bounds=(lb, ub), lsmr_tol='auto', verbose=0)
        s = np.clip(sol.x, 0.0, self.p.max_omega**2)
        Ti = self.p.kT * s
        Mz = self.p.kM * s * self.p.spin_dir
        Omega_cmd = np.sqrt(s)
        return Ti, Mz, Omega_cmd
    # ----------------- ODE -----------------

    def f(self,
          t: float,
          x: NDArray[np.float64],
          u: NDArray[np.float64],
          input_type: Literal["body_wrench"]
          ):
        """
        Continuous-time dynamics: \dot{x} = f(x, u).
        """
        r_I, v_I, q_BI, omega_B, Omega = self.unpack_state(x)
        R_BI = quat_euler.R_BI_from_q(q_BI)   # I->B
        R_IB = R_BI.T                         # B->I

        # Map wrench to rotor thrust/torques via mixer
        Ti, Mz, Omega_cmd = self._inputs_to_forces(u, x, input_type)

        T_cmd, tau_x_cmd, tau_y_cmd, tau_z_cmd = np.asarray(u, dtype=float).reshape(4,)

        # NOTE: For 'body_wrench' we apply ideal actuation (no motor lag).
        dOmega = np.zeros(4)

        # Net thrust in body (z up)
        F_B = np.array([0.0, 0.0, float(np.sum(Ti))])

        # Forces in inertial
        F_drag_I = -self.p.cd_lin * v_I
        F_g_I = np.array([0.0, 0.0, -self.p.g * self.p.mass])
        F_T_I = R_IB @ F_B
        a_I = (F_g_I + F_T_I + F_drag_I) / self.p.mass

        # Moments: lever arm τ = r × F (sum over rotors) + yaw reaction Mz_i
  
        tau_B = np.array([tau_x_cmd, tau_y_cmd, tau_z_cmd], dtype=float)


        # Rotational damping (linear)
        tau_B -= self.p.cd_rot * omega_B

        # Rigid body rotational dynamics: J ω_dot = τ - ω×(J ω)
        omega_cross_Jomega = np.cross(omega_B, self.J @ omega_B)
        domega = self.J_inv @ (tau_B - omega_cross_Jomega)

        # Quaternion kinematics
        dq = quat_euler.q_dot_from_body_rates(q_BI, omega_B)

        # Position/velocity kinematics
        dr = v_I
        dv = a_I

        return self.pack_state(dr, dv, dq, domega, dOmega)

    # ----------------- Sensors -----------------

    def imu(self,
            x: NDArray[np.float64],
            u: NDArray[np.float64],
            input_type: Literal["body_wrench"],
            accel_noise_std: float = 0.0,
            gyro_noise_std: float = 0.0) :
        """
        IMU-like outputs in body frame: specific force (acc + gravity) and gyro ω_B.
        """
        r_I, v_I, q_BI, omega_B, _ = self.unpack_state(x)
        R_BI = quat_euler.R_BI_from_q(q_BI); R_IB = R_BI.T
        Ti, _, _ = self._inputs_to_forces(u, x, input_type)
        F_B = np.array([0.0, 0.0, float(np.sum(Ti))])
        F_T_I = R_IB @ F_B
        F_g_I = np.array([0.0, 0.0, -self.p.g * self.p.mass])
        F_drag_I = -self.p.cd_lin * v_I
        a_I = (F_g_I + F_T_I + F_drag_I) / self.p.mass
        # Specific force in body: f_B = R_BI(a_I - g_I)
        g_I = np.array([0.0, 0.0, -self.p.g])
        f_B = R_BI @ (a_I - g_I)
        if accel_noise_std > 0:
            f_B = f_B + np.random.randn(3) * accel_noise_std
        gyro = omega_B.copy()
        if gyro_noise_std > 0:
            gyro = gyro + np.random.randn(3) * gyro_noise_std
        return {"accel_body": f_B, "gyro_body": gyro}

    def gps(self, x: NDArray[np.float64], pos_noise_std: float = 0.0, vel_noise_std: float = 0.0) :
        r_I, v_I, *_ = self.unpack_state(x)
        p = r_I + (np.random.randn(3) * pos_noise_std if pos_noise_std > 0 else 0.0)
        v = v_I + (np.random.randn(3) * vel_noise_std if vel_noise_std > 0 else 0.0)
        return {"pos_ENU": p, "vel_ENU": v}





