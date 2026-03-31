import numpy as np
import controller.pid as pid
from numpy.typing import NDArray

class PositionPI:
    """
    Position Controller (X, Y).
    Includes Anti-Windup AND Error Clamping to prevent overshoot on long flights.
    """
    def __init__(self, 
                 kp_di_x: pid.PIDGains, 
                 kp_di_y: pid.PIDGains, 
                 kp_di_z: pid.PIDGains, 
                 accel_limit: float = 6.0,  
                 i_limit: float = 0.05     
                 ):
        
        self.accel_max = accel_limit
        self.i_max = i_limit

        # PID objects
        self.pid_x = pid.PID(kp_di_x, -self.accel_max, self.accel_max, -self.i_max, self.i_max)
        self.pid_y = pid.PID(kp_di_y, -self.accel_max, self.accel_max, -self.i_max, self.i_max)
        self.pid_z = pid.PID(kp_di_z, -10.0, 10.0, -5.0, 5.0)


        self.error_clamp = 4.0 

    def reset(self, r_I: NDArray[np.float64], r_ref: NDArray[np.float64]):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

    def step(self, r_ref: NDArray[np.float64], r_I: NDArray[np.float64], v_I: NDArray[np.float64], dt: float):
        # 1. Calculate Raw Errors
        ex = r_ref[0] - r_I[0]
        ey = r_ref[1] - r_I[1]
        
        # 2. CLAMP ERRORS
        ex_clamped = np.clip(ex, -self.error_clamp, self.error_clamp)
        ey_clamped = np.clip(ey, -self.error_clamp, self.error_clamp)
        
        # 3. PID Step

        ax = self.pid_x.step(ex_clamped, dt, d_meas=v_I[0])
        ay = self.pid_y.step(ey_clamped, dt, d_meas=v_I[1])
        
        return np.array([ax, ay, 0.0])