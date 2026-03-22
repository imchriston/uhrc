from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import dynamics
from generate_data_sensors import get_lidar_scan, sample_forest
from uhrc_ctrl_h import UHRCController
import utils.quat_euler as quat_euler


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "checkpoints/uhrc_bestA*.pth"
STATS_PATH  = "checkpoints/norm_statsA*.npz"

NUM_OBS     = 4      # 4-8 in training — 4 for cleaner single tests
MAX_STEPS   = 1500    # match training STEPS
DT          = 0.01
GOAL_RADIUS = 0.2    # match training goal reach threshold

# Training distribution — model trained on ALL of these ranges.
# normal  : start x∈[-9,-6] y∈[-5,5]   goal x∈[6,9] y∈[-5,5]
# omni    : start/goal anywhere in [-10,10]²
# close   : start within 5m of goal, anywhere in arena
# recovery: start x∈[0,11]
# → safe eval range covers the full omni arena
TRAIN_X = (-10.0, 10.0)
TRAIN_Y = (-10.0, 10.0)


def _rk4_step(dyn, t: float, x: np.ndarray,
              u: np.ndarray, dt: float = DT) -> np.ndarray:
    def f(tt: float, xx: np.ndarray) -> np.ndarray:
        return dyn.f(tt, xx, u, "body_wrench")
    k1 = f(t,          x)
    k2 = f(t + 0.5*dt, x + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, x + 0.5*dt*k2)
    k4 = f(t + dt,     x +     dt*k3)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    q = x_next[6:10]
    x_next[6:10] = q / (np.linalg.norm(q) + 1e-12)
    return x_next


def run_eval(
    seed:    int | None        = None,
    start:   np.ndarray | None = None,
    goal:    np.ndarray | None = None,
    n_obs:   int               = NUM_OBS,
    verbose: bool              = True,
) -> tuple[np.ndarray, list, np.ndarray, bool]:
    """
    Run one evaluation episode.

    Returns: (path, obstacles, goal, success)
    """
    if seed is not None:
        np.random.seed(seed)

    params = dynamics.QuadrotorParams()
    dyn    = dynamics.QuadrotorDynamics(params)

    # Fresh controller every episode — carry must never persist across episodes
    ctrl = UHRCController(MODEL_PATH, STATS_PATH, device="cpu")
    ctrl.reset()

    # Default: sample from the full training arena
    if start is None:
        start = np.array([np.random.uniform(*TRAIN_X),
                          np.random.uniform(*TRAIN_Y), 0.0])
    if goal is None:
        goal  = np.array([np.random.uniform(*TRAIN_X),
                          np.random.uniform(*TRAIN_Y), 0.0])
        # Ensure minimum 3m separation
        while np.linalg.norm(goal[:2] - start[:2]) < 3.0:
            goal = np.array([np.random.uniform(*TRAIN_X),
                             np.random.uniform(*TRAIN_Y), 0.0])

    obstacles = sample_forest(n_obs, start[:2], goal[:2])
    goal_dist = float(np.linalg.norm(goal[:2] - start[:2]))

    x_curr = dyn.pack_state(
        start, np.zeros(3), np.array([1., 0., 0., 0.]),
        np.zeros(3), np.zeros(4),
    )

    if verbose:
        # Frame convention check
        r_I0, _, q_BI0, *_ = dyn.unpack_state(x_curr)
        R_BI0      = quat_euler.R_BI_from_q(q_BI0)
        goal_body0 = R_BI0 @ (goal - r_I0)
        print("── Frame diagnostic ──────────────────────────────────")
        print(f"  goal_world : {(goal - r_I0).round(3)}")
        print(f"  goal_body  : {goal_body0.round(3)}")
        print(f"  R_BI[0,0]  : {R_BI0[0,0]:.4f}  (expect +1 at zero yaw)")
        conv = "✅ OK" if goal_body0[0] > 0 else "❌ AXIS FLIP"
        print(f"  Convention : {conv}")
        print("──────────────────────────────────────────────────────")
        for i, (c, r) in enumerate(obstacles):
            print(f"  Obs {i}: center={np.array(c[:2]).round(2)}  r={float(r):.2f}m")
        print(f"🚀 start={start[:2].round(2)}  goal={goal[:2].round(2)}"
              f"  dist={goal_dist:.1f}m  obs={n_obs}")

    path  = []
    t     = 0.0
    r_new = start.copy()

    for step in range(MAX_STEPS):
        r_I, v_I, q_BI, w_B, Omega = dyn.unpack_state(x_curr)
        psi   = float(quat_euler.euler_from_q(q_BI)[2])
        lidar = get_lidar_scan(r_I, psi, obstacles, num_rays=32)

        u_nn, sub_nn = ctrl.get_action(r_I, v_I, q_BI, w_B, Omega, lidar, goal)

        if verbose and step % 100 == 0:
            d = np.linalg.norm(r_I[:2] - goal[:2])
            print(f"  step {step:3d}  pos=({r_I[0]:6.2f},{r_I[1]:6.2f})"
                  f"  Fz={u_nn[0]:.3f}"
                  f"  τ=({u_nn[1]:+.4f},{u_nn[2]:+.4f})"
                  f"  sub=({sub_nn[0]:+.2f},{sub_nn[1]:+.2f})"
                  f"  dist={d:.2f}m")

        path.append(r_I.copy())
        x_curr = _rk4_step(dyn, t, x_curr, u_nn)
        t     += DT

        r_new, *_ = dyn.unpack_state(x_curr)
        

        crashed = any(
            float(np.linalg.norm(r_new[:2] - np.asarray(c[:2]))) < float(r_obs)
            for c, r_obs in obstacles
        )
        reached = float(np.linalg.norm(r_new[:2] - goal[:2])) < GOAL_RADIUS

        if crashed:
            path.append(r_new.copy())
            if verbose:
                print(f"💥 Crash at step {step}  pos={r_new[:2].round(2)}")
            return np.array(path), obstacles, goal, False

        if reached:
            path.append(r_new.copy())
            if verbose:
                print(f"✅ Goal reached at step {step}  ({t:.2f}s)")
            return np.array(path), obstacles, goal, True

    final_dist = float(np.linalg.norm(r_new[:2] - goal[:2]))
    if verbose:
        print(f"⚠️  Timeout — final dist: {final_dist:.2f}m")
    return np.array(path), obstacles, goal, False


def plot_result(
    path:      np.ndarray,
    obstacles: list,
    goal:      np.ndarray,
    success:   bool,
    title:     str = "UHRC Forest Evaluation",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for c, r in obstacles:
        ax.add_patch(Circle(c[:2], r, color="tomato", alpha=0.5))

    colour = "royalblue" if success else "darkorange"
    label  = "UHRC Path (success)" if success else "UHRC Path (failed)"
    ax.plot(path[:, 0], path[:, 1], color=colour, linewidth=2, label=label)
    ax.plot(path[0, 0],  path[0, 1],  "go", markersize=10, label="Start")
    ax.plot(goal[0],     goal[1],     "b*", markersize=16, label="Goal",  zorder=5)
    ax.plot(path[-1, 0], path[-1, 1], "gs", markersize=10, label="End",   zorder=4)

    # Draw goal radius circle
    ax.add_patch(Circle(goal[:2], GOAL_RADIUS,
                        fill=False, linestyle="--", color="blue", alpha=0.4))

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.show()


def batch_eval(n: int = 20, seed_offset: int = 0) -> float:
    """Run N randomised episodes and report success rate."""
    results = []
    for i in range(n):
        _, _, _, ok = run_eval(seed=seed_offset + i, verbose=False)
        results.append(ok)
        print(f"  ep {i+1:3d}/{n}  {'✅' if ok else '❌'}", flush=True)
    rate = float(np.mean(results)) * 100.0
    print(f"\n  Result: {sum(results)}/{n} ({rate:.0f}%)")
    return rate


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        batch_eval(n)
    else:
        # Single eval — pass explicit start/goal or sample randomly
        # Example fixed scenario:
        #   run_eval(start=np.array([-8.,0.,0.]), goal=np.array([7.,1.,0.]))
        path, obs, goal, success = run_eval(seed=302844) #8 4
        #25541455, 777777
        plot_result(path, obs, goal, success)