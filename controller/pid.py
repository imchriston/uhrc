from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class PIDGains:
    kp: float
    ki: float =0.0 
    kd: float =0.0

class PID:
    """
    PID with:
      - anti-windup via integral clamping,
      - derivative low-pass filter (1st order),
      - output clamping.
    Call step(error, dt, d_meas=None) each control tick.
    """
    def __init__(self, gains: PIDGains, out_min: float = -np.inf, out_max: float =  np.inf,i_min: float = -np.inf,i_max: float =  np.inf,
                 d_cutoff_hz: float = 20.0):
        self.kp = gains.kp
        self.ki = gains.ki
        self.kd = gains.kd

        self.out_min = out_min
        self.out_max = out_max
        self.i_min = i_min
        self.i_max = i_max

        # state
        self.i_term = 0.0
        self.d_state = 0.0  #state derivative
        self.prev_err = 0.0
        self.initialized = False

        # derivative LPF coefficient (
        self.d_cutoff_hz = d_cutoff_hz

    def reset(self, value: float = 0.0):
        self.i_term = 0.0
        self.d_state = 0.0
        self.prev_err = value
        self.initialized = False

    def lpf(self, dt: float):
        # Bilinear 1st-order LPF: alpha ~ (2π f_c dt) / (1 + 2π f_c dt)
        if self.d_cutoff_hz <= 0.0:
            return 1.0
        rc = 1.0 / (2.0 * np.pi * self.d_cutoff_hz)
        alpha = dt / (rc + dt)
        return float(np.clip(alpha, 0.0, 1.0))

    def step(self, error: float, dt: float, d_meas: Optional[float] = None):
        """      if dt <= 0.0:
            # fail-safe: pure P
            return float(np.clip(self.kp * error, self.out_min, self.out_max))
        """
        # integral
        self.i_term += self.ki * error * dt
        self.i_term = float(np.clip(self.i_term, self.i_min, self.i_max))

        # derivative (filtered)
        if d_meas is not None:
            d_raw = -d_meas            # derivative on measurement: d(error)/dt ≈ -d(y)/dt
        else:
            if not self.initialized:
                d_raw = 0.0
                self.initialized = True
            else:
                d_raw = (error - self.prev_err) / dt

        alpha = self.lpf(dt)
        self.d_state = (1.0 - alpha) * self.d_state + alpha * d_raw

        # PID
        u = self.kp * error + self.i_term + self.kd * self.d_state
        u_sat = float(np.clip(u, self.out_min, self.out_max))

        # simple back-calculation anti-windup (if saturated, back off integral slightly)
        if self.ki > 0.0 and (u != u_sat):
            aw = u - u_sat
            # back-calc factor (tune small ~0.1..1.0)
            k_aw = 0.3
            self.i_term -= k_aw * aw

        self.prev_err = error
        return u_sat
