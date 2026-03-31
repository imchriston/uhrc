"""
generate_forest_data.py
═══════════════════════
Generates behavioural cloning data for UHRC forest navigation.

Expert policy: A* path planner on a 2D occupancy grid → velocity commands
               → PID attitude controller → body wrench actions.

Replaces the previous potential-field expert which:
  - Got stuck in local minima (tight gaps discarded as failures)
  - Produced wide detour paths (model learned suboptimal avoidance)
  - Systematically excluded difficult scenarios from training data

A* guarantees:
  - Finds the shortest collision-free path if one exists
  - Navigates tight gaps the potential field would miss
  - Consistent path quality across all obstacle configurations
  - Episodes only fail if truly no path exists (correctly excluded)
"""

import os
import heapq
import numpy as np
from tqdm import tqdm

import dynamics
import controller.pid as pid
import utils.quat_euler as quat_euler
import controller.position as position_control
import controller.attitude as angle_control

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_EPISODES   = 500 
STEPS          = 1500
DT             = 0.01

NUM_RAYS       = 32
FOV            = np.pi
LIDAR_RANGE    = 5.0
GOAL_RANGE     = 15.0
MAX_RANGE      = LIDAR_RANGE
NUM_OBSTACLES  = 8

OUTPUT_FILE    = "data/expert_data.npz"   # matches DATA_PATH in train_uhrc.py

V_MAX          = 2.0
HOVER_STEPS    = 150   # steps to hover at goal after reaching it (1.5s at DT=0.01)
REACH_RADIUS   = 0.5   # success radius for reaching the goal (metres) — episode ends when reached

# ── A* grid parameters 
GRID_RES       = 0.25    # metres per cell — finer = more accurate, slower
GRID_MARGIN    = 2.0     # padding around start/goal bounding box (metres)
INFLATE_RADIUS = 0.4     # robot body radius added to obstacle radii for safety


#  LiDAR 

LIDAR_ALT_GATE = 1.5

def get_lidar_scan(pos, yaw, obstacles, num_rays=32, fov=np.pi, max_range=5.0):
    ray_angles   = np.linspace(-0.5 * fov, 0.5 * fov, num_rays, dtype=np.float64)
    global_angles = yaw + ray_angles
    ray_vecs     = np.stack([np.cos(global_angles), np.sin(global_angles)], axis=1)

    ranges  = np.full((num_rays,), float(max_range), dtype=np.float64)
    pos_2d  = np.asarray(pos[:2], dtype=np.float64)
    drone_z = float(pos[2])

    for obs_pos, obs_r in obstacles:
        obs_z = float(obs_pos[2]) if len(obs_pos) > 2 else 0.0
        if abs(drone_z - obs_z) > LIDAR_ALT_GATE:
            continue
        c       = np.asarray(obs_pos[:2], dtype=np.float64)
        r       = float(obs_r)
        to_c    = c - pos_2d
        t_proj  = ray_vecs @ to_c
        to_c_sq = float(to_c @ to_c)
        perp_sq = to_c_sq - t_proj**2
        hit_mask = (perp_sq <= r * r) & (t_proj > 0.0)
        if not np.any(hit_mask):
            continue
        under    = np.maximum(r * r - perp_sq[hit_mask], 0.0)
        dt_hit   = np.sqrt(under)
        dist     = np.maximum(t_proj[hit_mask] - dt_hit, 0.0)
        ranges[hit_mask] = np.minimum(ranges[hit_mask], dist)

    return np.minimum(ranges, max_range).astype(np.float32)


#  A* PATH PLANNER

class AStarPlanner:
    """
    2D A* on a uniform grid built from circular obstacles.

    Usage:
        planner = AStarPlanner(obstacles, start_xy, goal_xy)
        path    = planner.plan()   # list of (x, y) waypoints, or None if no path
    """

    def __init__(self, obstacles, start_xy, goal_xy,
                 res=GRID_RES, margin=GRID_MARGIN, inflate=INFLATE_RADIUS):
        self.res     = res
        self.inflate = inflate

        # Grid bounds — tight bounding box around start/goal + margin
        xs = [start_xy[0], goal_xy[0]]
        ys = [start_xy[1], goal_xy[1]]
        for (c, r) in obstacles:
            xs += [c[0] - r - inflate, c[0] + r + inflate]
            ys += [c[1] - r - inflate, c[1] + r + inflate]

        self.x_min = min(xs) - margin
        self.x_max = max(xs) + margin
        self.y_min = min(ys) - margin
        self.y_max = max(ys) + margin

        self.nx = int(np.ceil((self.x_max - self.x_min) / res)) + 1
        self.ny = int(np.ceil((self.y_max - self.y_min) / res)) + 1

        # Build binary occupancy grid (True = occupied)
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)
        for (c, r) in obstacles:
            r_inf = r + inflate
            # Only iterate over cells that could be within radius
            ix_lo = max(0, int((c[0] - r_inf - self.x_min) / res))
            ix_hi = min(self.nx - 1, int((c[0] + r_inf - self.x_min) / res) + 1)
            iy_lo = max(0, int((c[1] - r_inf - self.y_min) / res))
            iy_hi = min(self.ny - 1, int((c[1] + r_inf - self.y_min) / res) + 1)
            for ix in range(ix_lo, ix_hi + 1):
                for iy in range(iy_lo, iy_hi + 1):
                    wx = self.x_min + ix * res
                    wy = self.y_min + iy * res
                    if (wx - c[0])**2 + (wy - c[1])**2 <= r_inf**2:
                        self.grid[ix, iy] = True

        self.start = self._world_to_grid(start_xy)
        self.goal  = self._world_to_grid(goal_xy)

    def _world_to_grid(self, xy):
        ix = int(round((xy[0] - self.x_min) / self.res))
        iy = int(round((xy[1] - self.y_min) / self.res))
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        return (ix, iy)

    def _grid_to_world(self, node):
        x = self.x_min + node[0] * self.res
        y = self.y_min + node[1] * self.res
        return np.array([x, y])

    def _heuristic(self, a, b):
        # Octile distance — exact for 8-connected grid with diagonal moves
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

    def _neighbors(self, node):
        ix, iy = node
        # 8-connected
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = ix + dx, iy + dy
                if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny:
                    if not self.grid[nx_, ny_]:
                        cost = np.sqrt(2) if (dx != 0 and dy != 0) else 1.0
                        yield (nx_, ny_), cost

    def plan(self):
        """
        Returns list of world-frame (x, y) waypoints from start to goal,
        or None if no collision-free path exists.
        """
        start, goal = self.start, self.goal

        # If start or goal is inside an obstacle, planning will fail 
        if self.grid[start[0], start[1]]:
            return None
        if self.grid[goal[0], goal[1]]:
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score   = {start: 0.0}
        f_score   = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self._grid_to_world(current))
                    current = came_from[current]
                path.append(self._grid_to_world(start))
                path.reverse()
                return path

            for neighbor, step_cost in self._neighbors(current):
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor]   = tentative_g
                    f_score[neighbor]   = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found


def smooth_path(path, iterations=3):
    """
    Simple path smoothing — iteratively replace each waypoint with the
    average of itself and its neighbours.
    """
    if len(path) <= 2:
        return path
    pts = np.array(path)
    for _ in range(iterations):
        smoothed = pts.copy()
        for i in range(1, len(pts) - 1):
            smoothed[i] = 0.25 * pts[i-1] + 0.5 * pts[i] + 0.25 * pts[i+1]
        pts = smoothed
    return [pts[i] for i in range(len(pts))]



#  PATH TRACKER — converts waypoint path to velocity command

class PathTracker:
    """
    Pure-pursuit style tracker.
    Finds the lookahead waypoint along the planned path and returns a
    velocity command toward it. Advances waypoints as they are reached.
    """

    def __init__(self, path, lookahead=1.0, v_max=V_MAX):
        self.path      = [np.array(p) for p in path]
        self.lookahead = lookahead
        self.v_max     = v_max
        self.idx       = 0   # current target waypoint index

    def get_velocity_command(self, pos_xy):
        pos = np.array(pos_xy[:2])

        # Advance waypoint index until lookahead waypoint is far enough ahead
        while self.idx < len(self.path) - 1:
            dist = np.linalg.norm(self.path[self.idx] - pos)
            if dist > self.lookahead:
                break
            self.idx += 1

        target = self.path[self.idx]
        to_target = target - pos
        dist = np.linalg.norm(to_target) + 1e-6

        # Scale speed: ramp down smoothly in last 2m, stop at goal
        dist_to_end = np.linalg.norm(self.path[-1] - pos)
        if dist_to_end < 0.3:
            speed = 0.0   # close enough — stop command
        elif dist_to_end < 2.0:
            speed = self.v_max * (dist_to_end / 2.0)   # linear ramp 0→v_max over 2m
            speed = max(speed, 0.1)
        else:
            speed = self.v_max

        v_xy = (to_target / dist) * speed
        return np.array([v_xy[0], v_xy[1], 0.0])

    def is_done(self, pos_xy, tol=0.5):
        return np.linalg.norm(np.array(pos_xy[:2]) - self.path[-1]) < tol


#  Forest sampling 

def sample_forest(num_obs, start_xy, goal_xy):
    obstacles = []
    p_vec     = goal_xy - start_xy
    p_len     = float(np.linalg.norm(p_vec))
    p_dir     = p_vec / p_len if p_len > 1e-3 else np.zeros(2)

    # Place half the obstacles along the direct path (works for any direction)
    num_forced = num_obs // 2
    for _ in range(num_forced):
        if p_len > 7.0:
            t              = np.random.uniform(3.5, p_len - 3.5)
            center_on_line = start_xy + p_dir * t
            jitter         = np.random.uniform(-0.3, 0.3, size=2)  # tighter jitter
            c = np.array([center_on_line[0] + jitter[0],
                          center_on_line[1] + jitter[1], 0.0])
            r = np.random.uniform(0.6, 1.2)   # larger — was 0.4-1.0
            obstacles.append((c, r))

    # Remaining obstacles — within 2m of path corridor
    x_lo = min(start_xy[0], goal_xy[0]) - 2.0
    x_hi = max(start_xy[0], goal_xy[0]) + 2.0
    y_lo = min(start_xy[1], goal_xy[1]) - 2.0
    y_hi = max(start_xy[1], goal_xy[1]) + 2.0

    for _ in range(num_obs - len(obstacles)):
        placed = False
        for _try in range(100):
            t      = np.random.uniform(0.15, 0.85)
            along  = start_xy + p_dir * (t * p_len)
            perp   = np.array([-p_dir[1], p_dir[0]])
            offset = np.random.uniform(-2.0, 2.0)   # tighter — was 3.0
            ox, oy = along + perp * offset
            ox = np.clip(ox, x_lo, x_hi)
            oy = np.clip(oy, y_lo, y_hi)
            r  = np.random.uniform(0.5, 1.0)         # larger — was 0.3-0.8
            c  = np.array([ox, oy, 0.0])
            if np.linalg.norm(c[:2] - start_xy) < (r + 1.5): continue
            if np.linalg.norm(c[:2] - goal_xy)  < (r + 1.5): continue
            too_close = any(
                np.linalg.norm(c[:2] - ec[:2]) < (r + er + 0.6)
                for ec, er in obstacles
            )
            if too_close: continue
            obstacles.append((c, r))
            placed = True
            break
        if not placed:
            pass

    return obstacles


#  Observation builder (unchanged from original)

def build_obs(r_I, v_I, q_BI, omega_B, goal, lidar_ranges, g_scalar):
    """Build 45-dim observation. Omega removed — always zero in simulator."""
    R_BI       = quat_euler.R_BI_from_q(q_BI).T
    v_B        = R_BI @ v_I
    g_B        = R_BI @ np.array([0.0, 0.0, -g_scalar])
    goal_rel_B = R_BI @ (goal - r_I)

    goal_dist      = float(np.linalg.norm(goal_rel_B)) + 1e-6
    goal_dist_norm = np.array(
        [min(goal_dist, GOAL_RANGE) / GOAL_RANGE], dtype=np.float32
    )
    if goal_dist > GOAL_RANGE:
        goal_rel_B = goal_rel_B / goal_dist * GOAL_RANGE

    obs = np.concatenate([
        v_B.astype(np.float32),                                                    # 3   [0:3]
        omega_B.astype(np.float32),                                                # 3   [3:6]
        g_B.astype(np.float32),                                                    # 3   [6:9]
        goal_rel_B.astype(np.float32),                                             # 3   [9:12]
        goal_dist_norm,                                                            # 1   [12:13]
        np.clip(lidar_ranges / LIDAR_RANGE, 0.0, 1.0).astype(np.float32),        # 32  [13:45]
    ])                                                                             # = 45
    assert obs.shape == (45,), f"Expected 45-dim obs, got {obs.shape}"
    return obs


#  Main

SAVE_INTERVAL = 500    
                        


def flush_to_disk(output_file, batch_obs, batch_actions, batch_subgoals,
                  batch_done, batch_episode_id, batch_step_id, batch_obs_packs,
                  flush_num):
    """
    Append a batch of successful episodes to OUTPUT_FILE.
    If the file already exists, loads it and concatenates; otherwise creates fresh.
    Clears the in-memory batch lists after saving.
    """
    new = {
        "obs":            np.array(batch_obs,        dtype=np.float32),
        "actions":        np.array(batch_actions,    dtype=np.float32),
        "subgoals":       np.array(batch_subgoals,   dtype=np.float32),
        "episode_id":     np.array(batch_episode_id, dtype=np.int32),
        "step_id":        np.array(batch_step_id,    dtype=np.int32),
        "done":           np.array(batch_done,       dtype=np.int8),
        "obstacles_xy_r": np.array(batch_obs_packs,  dtype=np.float32),
    }

    if os.path.exists(output_file):
        old = np.load(output_file)
        merged = {k: np.concatenate([old[k], new[k]]) for k in new}
    else:
        merged = new

    np.savez_compressed(
        output_file,
        obs            = merged["obs"],
        actions        = merged["actions"],
        subgoals       = merged["subgoals"],
        episode_id     = merged["episode_id"],
        step_id        = merged["step_id"],
        done           = merged["done"],
        obstacles_xy_r = merged["obstacles_xy_r"],
    )

    total_eps  = int(merged["episode_id"].max()) + 1
    total_trans = len(merged["obs"])
    print(f"\n  💾 Flush #{flush_num}: saved {total_eps:,} eps / {total_trans:,} transitions → {output_file}")


def run():
    Kpdi_roll  = pid.PIDGains(4.1602, 0.0, 2.0247)
    Kpdi_pitch = pid.PIDGains(4.1602, 0.0, 2.0247)
    Kpdi_yaw   = pid.PIDGains(0.9848, 0.0, 0.9542)
    Kpdi_x     = pid.PIDGains(2.9232, 0.9932, 2.5031)
    Kpdi_y     = pid.PIDGains(2.9232, 0.9932, 2.5031)
    Kpdi_z     = pid.PIDGains(153,    61,     135)

    params = dynamics.QuadrotorParams()
    dyn    = dynamics.QuadrotorDynamics(params)
    g      = getattr(getattr(dyn, "p", None), "g", 9.81)

    batch_obs, batch_actions, batch_subgoals = [], [], []
    batch_done, batch_episode_id             = [], []
    batch_step_id, batch_obs_packs           = [], []

    successful_episodes = 0   # global counter across all flushes
    no_path_count       = 0
    flush_count         = 0

    if os.path.exists(OUTPUT_FILE):
        old = np.load(OUTPUT_FILE)
        successful_episodes = int(old["episode_id"].max()) + 1
        print(f"📂 Resuming — existing file has {successful_episodes:,} episodes, "
              f"{len(old['obs']):,} transitions")

    print(f"🌲 Generating forest data (A* expert): episodes={NUM_EPISODES}, steps={STEPS}, dt={DT}")

    for ep in tqdm(range(NUM_EPISODES)):

        # ── Episode type sampling ─────────────────────────────────────────────
        # 40% normal   — x∈[-9,-6] → x∈[6,9], full forest (+X dominant)
        # 20% no-obs   — same range, empty space, pure goal-seeking
        # 10% close    — 1-5m from goal, any direction
        # 10% recovery — start past/beside goal, teaches re-acquisition
        # 20% omni     — random start/goal in full arena, any direction
        ep_type = np.random.choice(
            ["normal", "no_obs", "close", "recovery", "omni", "tight_gap"],
            p=[0.35, 0.15, 0.10, 0.10, 0.20, 0.10]
        )

        start_x = np.random.uniform(-9.0, -6.0)
        start_y = np.random.uniform(-2.0,  2.0)
        goal_x  = np.random.uniform( 6.0,  9.0)
        goal_y  = np.random.uniform(-2.0,  2.0)
        n_obs_this_ep = np.random.randint(4, NUM_OBSTACLES + 1)

        if ep_type == "close":
            goal_x  = np.random.uniform(-8.0, 8.0)
            goal_y  = np.random.uniform(-8.0, 8.0)
            angle   = np.random.uniform(0, 2 * np.pi)
            dist    = np.random.uniform(1.0, 5.0)
            start_x = goal_x + dist * np.cos(angle)
            start_y = goal_y + dist * np.sin(angle)
            n_obs_this_ep = np.random.randint(0, 3)
        elif ep_type == "no_obs":
            start_x = np.random.uniform(-9.0, -6.0)
            start_y = np.random.uniform(-5.0,  5.0)
            goal_x  = np.random.uniform( 6.0,  9.0)
            goal_y  = np.random.uniform(-5.0,  5.0)
            n_obs_this_ep = 0
        elif ep_type == "recovery":
            goal_x  = np.random.uniform( 2.0,  7.0)
            goal_y  = np.random.uniform(-4.0,  4.0)
            offset  = np.random.choice(["past", "side"])
            if offset == "past":
                start_x = goal_x + np.random.uniform(1.0, 4.0)
                start_y = goal_y + np.random.uniform(-2.0, 2.0)
            else:
                start_x = goal_x + np.random.uniform(-2.0, 2.0)
                start_y = goal_y + np.random.uniform(2.0, 5.0) * np.random.choice([-1, 1])
            n_obs_this_ep = np.random.randint(0, 3)
        elif ep_type == "omni":
            for _ in range(20):
                start_x = np.random.uniform(-10.0,  5.0)
                start_y = np.random.uniform( -8.0,  8.0)
                goal_x  = np.random.uniform( -5.0, 10.0)
                goal_y  = np.random.uniform( -8.0,  8.0)
                if np.linalg.norm([goal_x - start_x, goal_y - start_y]) >= 5.0:
                    break
            n_obs_this_ep = np.random.randint(0, NUM_OBSTACLES + 1)
        elif ep_type == "tight_gap":

            start_x = np.random.uniform(-9.0, -6.0)
            start_y = np.random.uniform(-2.0,  2.0)
            goal_x  = np.random.uniform( 6.0,  9.0)
            goal_y  = np.random.uniform(-2.0,  2.0)
            n_obs_this_ep = np.random.randint(1, 4)   
        else:  # normal
            start_x = np.random.uniform(-9.0, -6.0)
            start_y = np.random.uniform(-5.0,  5.0)
            goal_x  = np.random.uniform( 6.0,  9.0)
            goal_y  = np.random.uniform(-5.0,  5.0)
            n_obs_this_ep = np.random.randint(4, NUM_OBSTACLES + 1)

        start = np.array([start_x, start_y, 0.0], dtype=np.float64)
        goal  = np.array([goal_x,  goal_y,  0.0], dtype=np.float64)
        obstacles = sample_forest(n_obs_this_ep, start[:2], goal[:2])

        # Injects tight-gap pair for tight gap episodes 
 
        if ep_type == "tight_gap":
            p_vec  = goal[:2] - start[:2]
            p_len  = float(np.linalg.norm(p_vec))
            p_dir  = p_vec / p_len if p_len > 1e-3 else np.array([1.0, 0.0])
            perp   = np.array([-p_dir[1], p_dir[0]])
            # Place the gap wall at 40–60% along the path
            t_gap  = np.random.uniform(0.4, 0.6)
            mid    = start[:2] + p_dir * (t_gap * p_len)
            r_wall = np.random.uniform(0.6, 1.0)
            gap_half = np.random.uniform(0.5, 0.8)   # half-gap width
            # Two obstacle centers symmetrically offset from midpoint
            c1 = np.array([mid[0] + perp[0] * (r_wall + gap_half),
                           mid[1] + perp[1] * (r_wall + gap_half), 0.0])
            c2 = np.array([mid[0] - perp[0] * (r_wall + gap_half),
                           mid[1] - perp[1] * (r_wall + gap_half), 0.0])
            obstacles.append((c1, r_wall))
            obstacles.append((c2, r_wall))
        planner = AStarPlanner(obstacles, start[:2], goal[:2])
        path    = planner.plan()

        if path is None:
            no_path_count += 1
            continue   

        path    = smooth_path(path, iterations=5)
        tracker = PathTracker(path, lookahead=1.0, v_max=V_MAX)

        # Random initial yaw — model must learn to navigate from any heading,
        init_yaw = np.random.uniform(-np.pi, np.pi)
        cy, sy   = np.cos(init_yaw / 2), np.sin(init_yaw / 2)
        q_init   = np.array([cy, 0.0, 0.0, sy])   # yaw-only quaternion

        x_curr = dyn.pack_state(start, np.zeros(3), q_init,
                                 np.zeros(3), np.zeros(4))

        pos_ctrl  = position_control.PositionPI(Kpdi_x, Kpdi_y, Kpdi_z, accel_limit=6.0)
        att_ctrl  = angle_control.AttitudePID(params, Kpdi_roll, Kpdi_pitch, Kpdi_yaw, Kpdi_z)
        r_virtual = start.copy()
        t_curr    = 0.0

        ep_obs, ep_actions, ep_subgoals = [], [], []
        ep_done, ep_step_ids, ep_obs_packs = [], [], []

        crashed      = False
        reached      = False
        hover_count  = 0      
        in_hover     = False  

        for k in range(STEPS + HOVER_STEPS):
            r_I, v_I, q_BI, omega_B, Omega = dyn.unpack_state(x_curr)
            euler = quat_euler.euler_from_q(q_BI)
            psi   = float(euler[2])

            lidar_ranges = get_lidar_scan(r_I, psi, obstacles,
                                          num_rays=NUM_RAYS, fov=FOV, max_range=MAX_RANGE)

            dist_to_goal = float(np.linalg.norm((goal - r_I)[:2]))

            # Check if drone has entered goal radius 
            if dist_to_goal < REACH_RADIUS and not in_hover:
                in_hover = True
                reached  = True

            if in_hover:
                # Zero velocity command — teaches the model to hold position.
                v_cmd_world = np.zeros(3)
                hover_count += 1
            else:
                v_cmd_world = tracker.get_velocity_command(r_I[:2])

            R_BI      = quat_euler.R_BI_from_q(q_BI).T
            v_subgoal = (R_BI @ v_cmd_world).astype(np.float32)

            obs = build_obs(r_I, v_I, q_BI, omega_B, goal, lidar_ranges, g)

            if in_hover:
                r_virtual[:] = goal.copy()   # PID reference = goal position
            else:
                r_virtual[:2] += v_cmd_world[:2] * DT
                r_virtual[2]   = goal[2]

            accel_world = pos_ctrl.step(r_virtual, r_I, v_I, DT)
            a_world_vec = np.array([accel_world[0], accel_world[1], 0.0])
            a_body_vec  = R_BI @ a_world_vec

            phi_ref   = -a_body_vec[1] / g
            theta_ref =  a_body_vec[0] / g
            psi_ref   = 0.0

            refs     = {'phi': phi_ref, 'theta': theta_ref,
                        'psi': psi_ref, 'z': r_virtual[2]}
            u_expert = att_ctrl.step(x_curr, refs, DT)

            # ── Termination checks ────────────────────────────────────────────
            for (c_obs, r_obs) in obstacles:
                if np.linalg.norm(r_I[:2] - np.asarray(c_obs[:2])) <= float(r_obs):
                    crashed = True
                    break

            done = bool(
                crashed or
                (in_hover and hover_count >= HOVER_STEPS) or
                (k == STEPS + HOVER_STEPS - 1)
            )

            ep_obs.append(obs)
            ep_actions.append(u_expert.astype(np.float32))
            ep_subgoals.append(v_subgoal)
            ep_step_ids.append(np.int32(k))
            ep_done.append(np.int8(done))

            obs_pack = np.zeros((NUM_OBSTACLES, 3), dtype=np.float32)
            for i, (c, r) in enumerate(obstacles[:NUM_OBSTACLES]):
                obs_pack[i] = [c[0], c[1], r]
            ep_obs_packs.append(obs_pack)

            x_curr  = step_rk4(dyn, t_curr, x_curr, u_expert, DT)
            t_curr += DT

            # DAgger perturbations 
            if (not done) and (not in_hover) and (k > 20) and \
               (k % 50 == 0) and (np.random.rand() < 0.3):
                r_p, v_p, q_p, w_p, Om_p = dyn.unpack_state(x_curr)
                v_p  = v_p + np.random.randn(3) * 0.2
                dpsi = np.random.uniform(-0.15, 0.15)
                dq   = np.array([np.cos(dpsi/2), 0., 0., np.sin(dpsi/2)])
                w0,x0,y0,z0 = q_p
                w1,x1,y1,z1 = dq
                q_p  = np.array([
                    w1*w0 - x1*x0 - y1*y0 - z1*z0,
                    w1*x0 + x1*w0 + y1*z0 - z1*y0,
                    w1*y0 - x1*z0 + y1*w0 + z1*x0,
                    w1*z0 + x1*y0 - y1*x0 + z1*w0,
                ])
                q_p    = q_p / (np.linalg.norm(q_p) + 1e-12)
                x_curr = dyn.pack_state(r_p, v_p, q_p, w_p, Om_p)

            if done:
                break


        # Checks every ~100 steps that:
        #   1. dist-to-goal is decreasing overall (drone making progress)
        #   2. subgoal speed near goal is low (stopping behaviour present)
        #   3. episode reached goal (not just timed out approaching it)
        episode_ok = reached and not crashed

        if episode_ok and len(ep_obs) > 200:
            ep_obs_arr  = np.array(ep_obs)
            ep_sub_arr  = np.array(ep_subgoals)

            # Check 1: dist-to-goal decreases over time
            dist_samples = ep_obs_arr[::100, 12] * GOAL_RANGE
            n_samples    = len(dist_samples)
            if n_samples >= 3:
                # Dist at end should be less than dist at start
                if dist_samples[-1] > dist_samples[0] * 0.8:
                    episode_ok = False   # drone didn't make meaningful progress

            # Check 2: subgoal speed drops near goal
            if reached and len(ep_sub_arr) > HOVER_STEPS:
                hover_subs  = ep_sub_arr[-HOVER_STEPS:]
                hover_speed = float(np.linalg.norm(hover_subs[:, :2], axis=1).mean())
                if hover_speed > 0.3:
                    episode_ok = False

        # Only keep episodes that passed 
        if episode_ok:
            n = len(ep_obs)
            batch_obs.extend(ep_obs)
            batch_actions.extend(ep_actions)
            batch_subgoals.extend(ep_subgoals)
            batch_done.extend(ep_done)
            batch_step_id.extend(ep_step_ids)
            batch_episode_id.extend([np.int32(successful_episodes)] * n)
            batch_obs_packs.extend(ep_obs_packs)
            successful_episodes += 1

            if successful_episodes % SAVE_INTERVAL == 0:
                flush_count += 1
                flush_to_disk(OUTPUT_FILE,
                              batch_obs, batch_actions, batch_subgoals,
                              batch_done, batch_episode_id, batch_step_id,
                              batch_obs_packs, flush_count)
                batch_obs.clear();     batch_actions.clear()
                batch_subgoals.clear(); batch_done.clear()
                batch_episode_id.clear(); batch_step_id.clear()
                batch_obs_packs.clear()

    if batch_obs:
        flush_count += 1
        flush_to_disk(OUTPUT_FILE,
                      batch_obs, batch_actions, batch_subgoals,
                      batch_done, batch_episode_id, batch_step_id,
                      batch_obs_packs, flush_count)

    final = np.load(OUTPUT_FILE)
    print(f"\n✅ Done — {OUTPUT_FILE}")
    print(f"   Successful episodes:  {successful_episodes:,} / {NUM_EPISODES:,}")
    print(f"   No-path skipped:      {no_path_count:,}")
    print(f"   Total transitions:    {len(final['obs']):,}")
    print(f"   obs shape:            {final['obs'].shape}")
    print(f"   Flushes performed:    {flush_count}")


def step_rk4(dyn, t, x, u, dt):
    def f(tt, xx): return dyn.f(tt, xx, u, "body_wrench")
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt,       x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    q = x_next[6:10]
    x_next[6:10] = q / (np.linalg.norm(q) + 1e-12)
    return x_next


if __name__ == "__main__":
    run()