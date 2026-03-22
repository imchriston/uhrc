
from __future__ import annotations
import math
from typing import Callable, Tuple, Optional, Dict, Any
import numpy as np
from scipy.integrate import solve_ivp
from numpy import sin, cos
from numpy.linalg import norm
from numpy.typing import NDArray



#-----------Quaternion-Euler Helper functions--------
def q_normalize(q: NDArray[np.float64]):
    """Normalize quaternion (w, x, y, z)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def q_mul(q1: NDArray[np.float64], q2: NDArray[np.float64]):
    """Quaternion product: q = q1 ⊗ q2 (Hamilton).
    Used to complete rotations of two axes"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def q_from_omega(omega_B: NDArray[np.float64]):
    """Maps angular velocity into pure-imaginary quaternion"""
    return np.array([0.0, omega_B[0], omega_B[1], omega_B[2]])


def q_dot_from_body_rates(q_BI: NDArray[np.float64], omega_B: NDArray[np.float64]):
    r"""
    Quaternion kinematics:  \dot{q} = 0.5 * q ⊗ Ω, with Ω = (0, ω_B).
    q_BI rotates vectors from I to B (body-from-inertial).
    """
    return 0.5 * q_mul(q_BI, q_from_omega(omega_B))


def R_BI_from_q(q_BI: NDArray[np.float64]):
    """Rotation matrix R_BI from quaternion (w, x, y, z) (maps inertial→body)."""
    w, x, y, z = q_BI
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    # Standard DCM
    return np.array([
        [ww + xx - yy - zz, 2*(xy + wz),      2*(xz - wy)],
        [2*(xy - wz),       ww - xx + yy - zz, 2*(yz + wx)],
        [2*(xz + wy),       2*(yz - wx),      ww - xx - yy + zz]
    ])

def R_IB_from_q(q_BI):
    """Rotation matrix mapping body → inertial"""
    return R_BI_from_q(q_BI).T


def euler_from_q(q_BI: NDArray[np.float64]) :
    """
    Extracts euler angles to be logged.
    Returns (roll φ about x_E, pitch θ about y_N, yaw ψ about z_U).
    """
    R_BI = R_BI_from_q(q_BI)
    R_IB = R_BI.T
    # ZYX from R_IB
    sy = -R_IB[2, 0]

    theta = math.asin(sy)
    psi = math.atan2(R_IB[1, 0], R_IB[0, 0])
    phi = math.atan2(R_IB[2, 1], R_IB[2, 2])
  
    return (phi, theta, psi)


def q_from_euler(phi: float, theta: float, psi: float) :
    """Quaternion from ZYX Euler angles (roll, pitch, yaw) for ENU."""
    c1, s1 = math.cos(psi/2), math.sin(psi/2)     # yaw
    c2, s2 = math.cos(theta/2), math.sin(theta/2) # pitch
    c3, s3 = math.cos(phi/2), math.sin(phi/2)     # roll
   
    w = c1*c2*c3 + s1*s2*s3
    x = c1*c2*s3 - s1*s2*c3
    y = c1*s2*c3 + s1*c2*s3
    z = s1*c2*c3 - c1*s2*s3
    q_IB = np.array([w, x, y, z])
    q_BI = q_IB.copy()
    q_BI[1:] *= -1.0
    return q_normalize(q_BI)



