"""Multi-seed comparison on randomly-generated friction terrain.

The ground stays flat (low fall risk) but its friction varies in space: each
seed gets a random number of friction zones (ice / slick / normal / grip /
rubber) of random length, placed within the robot's travel range. The ground's
lateral friction is updated each simulation step from the robot's forward
position (methods.terrain.apply_dynamic_friction), so the robot must adapt its
gait as it crosses surfaces of different grip. All three methods run on the
same terrain per seed; results are averaged across seeds.

A fresh subprocess is used per seed so the environment is rebuilt cleanly.

Usage (from repo root):
    python experiment-friction/run_multiseed.py                 # all seeds + figure
    python experiment-friction/run_multiseed.py --seed 0        # one seed (worker)
    python experiment-friction/run_multiseed.py --trials 100 --seeds 5
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

RESULTS_DIR = os.path.join(_HERE, "results")   # writes into this experiment folder
TARGET_FORWARD_POSITION = 4.0
ROBOT_MASS = 10.0
N_INIT = 5

# Cautious MARX-EFE config with per-cycle online adaptation (re-selects every
# UPDATE_EVERY steps so it can react as it crosses friction zones mid-episode).
TARGET_VELOCITY     = 1.0
GOAL_PRIOR_STD      = (np.sqrt(0.5), np.sqrt(0.5), np.deg2rad(45), np.deg2rad(45))
CONTROL_PRIOR_SCALE = 0.15
UPDATE_EVERY        = 100     # re-select actions every 100 steps (= 1 s)
RAMP_STEPS          = 20
FORGETTING          = 1.0
TIME_HORIZON        = 2       # EFE planning horizon (steps)

METHOD_COLORS = {"GridSearch": "tab:green", "BO": "tab:blue", "MARXEFE": "tab:orange"}


PARAM_COLS = ["couplinggain", "wswing", "wstance", "FFAST",
              "STOPGAIN", "hipamplitude", "kneeamplitude", "b"]


def _best_params(method, seed):
    import pandas as pd
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"{method}_seed{seed}.csv"))
    row = df.loc[df["J"].idxmax()]
    return np.array([row[c] for c in PARAM_COLS], dtype=float)


def run_one_seed(seed, n_trials):
    from methods import terrain
    from methods.cpg_bounds import bounds
    from methods.grid_search import gridsearch_optimize_cpg
    from methods.bo_optimizer import bo_optimize_cpg, run_cpg_trial
    from methods import marxefe_optimizer as mx

    cfg = terrain.sample_friction(seed)
    terrain.TERRAIN_CONFIG = cfg
    zones = [(z[0], z[2]) for z in cfg["zones"]]
    print(f"[seed {seed}] friction terrain: n_zones={cfg['n_zones']} "
          f"zones={zones}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Evaluation traces (t, vx, y) per method, captured right after each method
    # so the correct PyBullet client is active when we replay the best gait.
    traces = {}

    gridsearch_optimize_cpg(
        bounds, target_forward_position=TARGET_FORWARD_POSITION,
        robot_mass=ROBOT_MASS, n_trials=n_trials,
        optimizer_name="GridSearch", seed=seed, results_dir=RESULTS_DIR)
    td = run_cpg_trial(_best_params("GridSearch", seed),
                       TARGET_FORWARD_POSITION, robot_mass=ROBOT_MASS)
    traces["GridSearch"] = (td["t"], td["vx"], td["base_pos"][:, 1])

    bo_optimize_cpg(
        bounds, target_forward_position=TARGET_FORWARD_POSITION,
        robot_mass=ROBOT_MASS, n_trials=n_trials, n_init=N_INIT,
        optimizer_name="BO", seed=seed, results_dir=RESULTS_DIR)
    td = run_cpg_trial(_best_params("BO", seed),
                       TARGET_FORWARD_POSITION, robot_mass=ROBOT_MASS)
    traces["BO"] = (td["t"], td["vx"], td["base_pos"][:, 1])

    mx.marxefe_optimize_cpg(
        bounds, target_forward_position=TARGET_FORWARD_POSITION,
        robot_mass=ROBOT_MASS, n_trials=n_trials,
        optimizer_name="MARXEFE", seed=seed, results_dir=RESULTS_DIR,
        goal_prior_std=GOAL_PRIOR_STD, control_prior_scale=CONTROL_PRIOR_SCALE,
        target_velocity=TARGET_VELOCITY, update_every=UPDATE_EVERY,
        ramp_steps=RAMP_STEPS, forgetting=FORGETTING,
        time_horizon=TIME_HORIZON, debug_first_trial=False)
    agent = mx._last_agent
    agent.reset_buffer()
    td = mx.run_episode_maxrefe(
        agent, mx.quadruped, mx.joint_IDs_full, mx.filtered_joint_IDs,
        mx.feet_joint_IDs, dt=0.01, episode_length=4.5, lambda_energy=1e-2,
        target_forward_position=TARGET_FORWARD_POSITION,
        update_every=UPDATE_EVERY, ramp_steps=RAMP_STEPS)
    traces["MARXEFE"] = (td["t"], td["vx"], td["base_pos"][:, 1])

    # Save traces + zone boundaries (and their friction values, so the recovery
    # analysis can identify friction *drops*) for the transition analysis.
    save = {"zones_y": np.array([z[0] for z in cfg["zones"]], dtype=float),
            "zones_mu": np.array([z[1] for z in cfg["zones"]], dtype=float),
            "base_mu": float(cfg["base_mu"]),
            "target_v": TARGET_VELOCITY}
    for m, (t, vx, y) in traces.items():
        save[f"{m}_t"] = np.asarray(t)
        save[f"{m}_vx"] = np.asarray(vx)
        save[f"{m}_y"] = np.asarray(y)
    np.savez(os.path.join(RESULTS_DIR, f"traces_seed{seed}.npz"), **save)
    print(f"[seed {seed}] saved evaluation traces.", flush=True)


def aggregate_and_plot(seeds, n_trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from methods import terrain

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n=== terrain per seed ===")
    for s in seeds:
        cfg = terrain.sample_friction(s)
        print(f"  seed {s}: {cfg['n_zones']} zones -> "
              f"{[z[2] for z in cfg['zones']]}")

    fig, ax = plt.subplots(figsize=(9, 6))
    print("\n=== summary (mean ± std over seeds) ===")
    for method, color in METHOD_COLORS.items():
        curves, best_finals, fall_rates = [], [], []
        for s in seeds:
            path = os.path.join(RESULTS_DIR, f"{method}_seed{s}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path).sort_values("trial")
            best = np.maximum.accumulate(df["J"].values)[:n_trials]
            if len(best) < n_trials:
                best = np.pad(best, (0, n_trials - len(best)), mode="edge")
            curves.append(best)
            best_finals.append(best[-1])
            fall_rates.append(100.0 * df["fell"].mean())
        if not curves:
            continue
        C = np.vstack(curves)
        mean, std = C.mean(0), C.std(0)
        x = np.arange(1, n_trials + 1)
        ax.plot(x, mean, "-", color=color, label=method)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        print(f"  {method:11s} final best J = {np.mean(best_finals):.2f} "
              f"± {np.std(best_finals):.2f}   "
              f"fall rate = {np.mean(fall_rates):.0f} ± {np.std(fall_rates):.0f} %")

    ax.axhline(1.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_ylim(-1.0, 3.0)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best-so-far objective J")
    ax.set_title(f"Friction terrain (random ice/rubber zones) — best-so-far J, "
                 f"mean ± std over {len(seeds)} seeds")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    out = os.path.join(fig_dir, "comparison_friction.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="worker mode: one seed")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds (0..N-1)")
    args = ap.parse_args()

    if args.seed is not None:
        run_one_seed(args.seed, args.trials)
        return

    seeds = list(range(args.seeds))
    t0 = time.time()
    for s in seeds:
        print(f"\n########## SEED {s} / {seeds[-1]} ##########", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--seed", str(s), "--trials", str(args.trials)],
                       check=True, cwd=_REPO_ROOT)
    print(f"\nAll seeds done in {(time.time()-t0)/60:.1f} min.")
    aggregate_and_plot(seeds, args.trials)


if __name__ == "__main__":
    main()
