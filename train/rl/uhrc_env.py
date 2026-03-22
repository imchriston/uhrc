"""
uhrc_env.py — Forest environment for TD3 fine-tuning of UHRC
═════════════════════════════════════════════════════════════
obs_dim = 45  (Omega removed — always zero)

Fixes over previous version:
  - GOAL_RANGE=15.0 used for goal_dist_norm (was MAX_RANGE=5.0 — wrong)
  - obs_std clamped to min 1e-3
  - reset() samples normal + omni + close episode types
  - proximity shaping: only boosts *closing* progress, not free reward for
    staying inside 3m (that created hover-short local optimum)
  - hover reward weight 1.0 (was 10.0 — dominated all other signals)
  - duplicate proximity block removed (was applied twice)
  - _sample_forest: parametric placement works for any travel direction
"""
from __future__ import annotations
import numpy as np
import dynamics
import utils.quat_euler as quat_euler

DT          = 0.01
NUM_RAYS    = 32
FOV         = np.pi
MAX_RANGE   = 5.0    # lidar sensor range — matches LIDAR_RANGE in generator
GOAL_RANGE  = 15.0   # goal_rel_B clip — matches GOAL_RANGE in generator
GOAL_RADIUS = 1.5
MAX_STEPS   = 2000   # increased from 1500 — 15m at V_MAX=2m/s needs ~750 steps
                     # A* detours around obstacles easily reach 1200-1500 steps
                     # 1500 was causing legitimate navigating episodes to timeout


def _get_lidar(pos, yaw, obstacles, num_rays=32, fov=np.pi, max_range=5.0):
    ray_angles    = np.linspace(-0.5*fov, 0.5*fov, num_rays, dtype=np.float64)
    global_angles = yaw + ray_angles
    ray_vecs      = np.stack([np.cos(global_angles), np.sin(global_angles)], axis=1)
    ranges        = np.full((num_rays,), float(max_range), dtype=np.float64)
    pos_2d        = np.asarray(pos[:2], dtype=np.float64)
    for obs_pos, obs_r in obstacles:
        c       = np.asarray(obs_pos[:2], dtype=np.float64)
        r       = float(obs_r)
        to_c    = c - pos_2d
        t_proj  = ray_vecs @ to_c
        to_c_sq = float(to_c @ to_c)
        perp_sq = to_c_sq - t_proj**2
        hit     = (perp_sq <= r*r) & (t_proj > 0.0)
        if not np.any(hit): continue
        under   = np.maximum(r*r - perp_sq[hit], 0.0)
        dist    = np.maximum(t_proj[hit] - np.sqrt(under), 0.0)
        ranges[hit] = np.minimum(ranges[hit], dist)
    return np.minimum(ranges, max_range).astype(np.float32)


def _sample_forest(n_obs, start_xy, goal_xy, rng):
    """Parametric obstacle placement — works for any travel direction."""
    obstacles = []
    p_vec = goal_xy - start_xy
    p_len = float(np.linalg.norm(p_vec))
    p_dir = p_vec / p_len if p_len > 1e-3 else np.zeros(2)
    for _ in range(n_obs):
        for _ in range(100):
            t     = rng.uniform(0.2, 0.8)
            along = start_xy + p_dir * (t * p_len)
            perp  = np.array([-p_dir[1], p_dir[0]])
            off   = rng.uniform(-2.0, 2.0)
            c     = np.array([along[0] + perp[0]*off,
                               along[1] + perp[1]*off, 0.0])
            r     = rng.uniform(0.5, 1.0)
            if np.linalg.norm(c[:2] - start_xy) < r + 1.5: continue
            if np.linalg.norm(c[:2] - goal_xy)  < r + 1.5: continue
            too_close = any(
                np.linalg.norm(c[:2] - ec[:2]) < r + er + 0.6
                for ec, er in obstacles
            )
            if too_close: continue
            obstacles.append((c, r))
            break
    return obstacles


def _step_rk4(dyn, t, x, u):
    def f(tt, xx): return dyn.f(tt, xx, u, "body_wrench")
    k1 = f(t,          x)
    k2 = f(t + 0.5*DT, x + 0.5*DT*k1)
    k3 = f(t + 0.5*DT, x + 0.5*DT*k2)
    k4 = f(t + DT,     x +     DT*k3)
    xn = x + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    q  = xn[6:10]; xn[6:10] = q / (np.linalg.norm(q) + 1e-12)
    return xn


def _build_obs(r_I, v_I, q_BI, omega_B, goal, lidar, g):
    R_BI       = quat_euler.R_BI_from_q(q_BI).T
    v_B        = R_BI @ v_I
    g_B        = R_BI @ np.array([0., 0., -g])
    goal_rel_B = R_BI @ (goal - r_I)
    dist       = float(np.linalg.norm(goal_rel_B)) + 1e-6
    # Use GOAL_RANGE (15m) not MAX_RANGE (5m) — gives gradient over full trajectory
    dist_norm  = np.array([min(dist, GOAL_RANGE) / GOAL_RANGE], dtype=np.float32)
    if dist > GOAL_RANGE:
        goal_rel_B = goal_rel_B / dist * GOAL_RANGE
    return np.concatenate([
        v_B.astype(np.float32),
        omega_B.astype(np.float32),
        g_B.astype(np.float32),
        goal_rel_B.astype(np.float32),
        dist_norm,
        np.clip(lidar / MAX_RANGE, 0., 1.).astype(np.float32),
    ])  # 45-dim


class ForestEnv:
    """
    45-dim obs environment for TD3 fine-tuning.

    Reward:
      r_progress   3.0 × (prev_dist - curr_dist)         dense
      r_proximity  1.5 × closing progress if dist < 3m   amplified near-goal
      r_hover      1.0 × (1 - speed/0.5) if dist < 2.25m stops drift-past
      r_obstacle   −0.3 × proximity if within 1.5m        soft penalty
      r_time       −0.02 per step                         efficiency
      r_success    +100 terminal                          reached goal
      r_crash      −50  terminal                          collision
    """

    obs_dim = 45
    act_dim = 4

    def __init__(self, obs_mean: np.ndarray, obs_std: np.ndarray,
                 num_obstacles: int = 6, seed: int | None = None):
        self.obs_mean      = obs_mean
        self.obs_std       = np.where(obs_std < 1e-3, 1e-3, obs_std)
        self.num_obstacles = num_obstacles
        self._rng          = np.random.default_rng(seed)

        params    = dynamics.QuadrotorParams()
        self._dyn = dynamics.QuadrotorDynamics(params)
        self._g   = getattr(getattr(self._dyn, "p", None), "g", 9.81)

        self._x_curr:    np.ndarray
        self._goal:      np.ndarray
        self._obstacles: list
        self._prev_dist: float
        self._t:         float
        self._step:      int

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ep_type = self._rng.choice(["normal", "omni", "close"],
                                   p=[0.60, 0.30, 0.10])

        if ep_type == "close":
            gx = self._rng.uniform(-8., 8.)
            gy = self._rng.uniform(-8., 8.)
            a  = self._rng.uniform(0, 2*np.pi)
            d  = self._rng.uniform(2., 6.)
            sx, sy = gx + d*np.cos(a), gy + d*np.sin(a)
        elif ep_type == "omni":
            for _ in range(20):
                sx = self._rng.uniform(-10., 5.)
                sy = self._rng.uniform(-8.,  8.)
                gx = self._rng.uniform(-5., 10.)
                gy = self._rng.uniform(-8.,  8.)
                if np.linalg.norm([gx-sx, gy-sy]) >= 5.: break
        else:  # normal
            sx = self._rng.uniform(-9., -6.)
            sy = self._rng.uniform(-5.,  5.)
            gx = self._rng.uniform( 6.,  9.)
            gy = self._rng.uniform(-5.,  5.)

        self._goal = np.array([gx, gy, 0.], dtype=np.float64)
        start      = np.array([sx, sy, 0.], dtype=np.float64)
        self._obstacles = _sample_forest(
            self.num_obstacles, start[:2], self._goal[:2], self._rng
        )
        self._x_curr = self._dyn.pack_state(
            start, np.zeros(3), np.array([1.,0.,0.,0.]),
            np.zeros(3), np.zeros(4)
        )
        self._t         = 0.
        self._step      = 0
        self._prev_dist = float(np.linalg.norm(start[:2] - self._goal[:2]))
        return self._obs()

    def step(self, action: np.ndarray, subgoal: np.ndarray | None = None):
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            [0., -0.5, -0.5, -0.5],
            [20.,  0.5,  0.5,  0.5],
        )

        r_I, v_I, q_BI, omega_B, _ = self._dyn.unpack_state(self._x_curr)
        curr_dist = float(np.linalg.norm(r_I[:2] - self._goal[:2]))

        reward = 0.0

        # 1. Dense progress
        reward += 3.0 * (self._prev_dist - curr_dist)

        # 2. Proximity shaping — amplify closing progress only (not free reward)
        if curr_dist < 3.0:
            progress_boost = 1.5 * (self._prev_dist - curr_dist)
            reward += max(progress_boost, 0.0)

        # 3. Hover reward — discourages flying through the goal
        if curr_dist < GOAL_RADIUS * 1.5:
            speed = float(np.linalg.norm(v_I[:2]))
            reward += 1.0 * max(0.0, 1.0 - speed / 0.5)

        # 4. Obstacle proximity penalty
        for obs_pos, obs_r in self._obstacles:
            d = float(np.linalg.norm(r_I[:2] - obs_pos[:2])) - float(obs_r)
            if d < 1.5:
                reward -= 0.3 * (1.5 - max(d, 0.0)) / 1.5

        # 5. Time penalty — reduced and distance-adaptive
        # Flat -0.02 over 2000 steps = -40 raw = -0.40 scaled, too harsh.
        # Far from goal: small penalty (drone should keep moving)
        # Near goal: larger penalty (drone should stop quickly)
        dist_factor = min(curr_dist / 5.0, 1.0)   # 1.0 far, 0.0 at goal
        reward -= 0.005 + 0.01 * (1.0 - dist_factor)  # -0.005 far, -0.015 near goal

        # ── Physics step ──────────────────────────────────────────────────────
        self._x_curr = _step_rk4(self._dyn, self._t, self._x_curr, action)
        self._t     += DT
        self._step  += 1

        if not np.isfinite(self._x_curr).all():
            info = {"reached": False, "crashed": True, "timeout": False,
                    "dist_to_goal": curr_dist}
            return self._obs_safe(), float(reward - 50.), True, info

        r_new, *_ = self._dyn.unpack_state(self._x_curr)
        crashed = any(
            float(np.linalg.norm(r_new[:2] - np.asarray(c[:2]))) < float(r)
            for c, r in self._obstacles
        )
        reached = float(np.linalg.norm(r_new[:2] - self._goal[:2])) < GOAL_RADIUS
        timeout = self._step >= MAX_STEPS

        if reached: reward += 100.
        if crashed: reward -= 50.
        done = bool(reached or crashed or timeout)
        self._prev_dist = float(np.linalg.norm(r_new[:2] - self._goal[:2]))

        info = {"reached": reached, "crashed": crashed, "timeout": timeout,
                "dist_to_goal": self._prev_dist}
        return self._obs(), float(reward), done, info

    def _obs(self) -> np.ndarray:
        r_I, v_I, q_BI, omega_B, _ = self._dyn.unpack_state(self._x_curr)
        psi   = float(quat_euler.euler_from_q(q_BI)[2])
        lidar = _get_lidar(r_I, psi, self._obstacles)
        raw   = _build_obs(r_I, v_I, q_BI, omega_B, self._goal, lidar, self._g)
        return ((raw - self.obs_mean) / self.obs_std).astype(np.float32)

    def _obs_safe(self) -> np.ndarray:
        return np.zeros(self.obs_dim, dtype=np.float32)