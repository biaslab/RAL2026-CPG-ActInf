"""Multi-seed comparison on randomly-generated multi-slope terrain.

Each seed gets a terrain with a randomly sampled number of slope segments
(1-4), each of random length, angle and sign (see methods.terrain.sample_
multislope). All three methods are run on the *same* terrain per seed, so the
comparison is fair within a seed and averaged across seeds for error bars.

A fresh subprocess is used per seed so the PyBullet environment (and its
terrain) is rebuilt cleanly each time.

Usage (from repo root):
    python experiment-sloped/run_multiseed.py                 # all seeds + figure
    python experiment-sloped/run_multiseed.py --seed 0        # one seed (worker)
    python experiment-sloped/run_multiseed.py --trials 100 --seeds 5
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

# Cautious MARX-EFE config (tuned on the single slope).
TARGET_VELOCITY     = 1.0
GOAL_PRIOR_STD      = (np.sqrt(0.5), np.sqrt(0.5), np.deg2rad(45), np.deg2rad(45))
CONTROL_PRIOR_SCALE = 0.15
UPDATE_EVERY        = 100     # re-select actions every 100 steps (= 1 s)
RAMP_STEPS          = 20
FORGETTING          = 1.0
TIME_HORIZON        = 2       # EFE planning horizon (steps)

METHOD_COLORS = {"GridSearch": "tab:green", "BO": "tab:blue", "MARXEFE": "tab:orange"}


def run_one_seed(seed, n_trials):
    from methods import terrain
    from methods.cpg_bounds import bounds
    from methods.grid_search import gridsearch_optimize_cpg
    from methods.bo_optimizer import bo_optimize_cpg
    from methods.marxefe_optimizer import marxefe_optimize_cpg

    cfg = terrain.sample_multislope(seed)
    terrain.TERRAIN_CONFIG = cfg
    print(f"[seed {seed}] terrain: n_slopes={cfg['n_slopes']} "
          f"segments={cfg['segments']}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gridsearch_optimize_cpg(
        bounds, target_forward_position=TARGET_FORWARD_POSITION,
        robot_mass=ROBOT_MASS, n_trials=n_trials,
        optimizer_name="GridSearch", seed=seed, results_dir=RESULTS_DIR)

    bo_optimize_cpg(
        bounds, target_forward_position=TARGET_FORWARD_POSITION,
        robot_mass=ROBOT_MASS, n_trials=n_trials, n_init=N_INIT,
        optimizer_name="BO", seed=seed, results_dir=RESULTS_DIR)

    marxefe_optimize_cpg(
        bounds, target_forward_position=TARGET_FORWARD_POSITION,
        robot_mass=ROBOT_MASS, n_trials=n_trials,
        optimizer_name="MARXEFE", seed=seed, results_dir=RESULTS_DIR,
        goal_prior_std=GOAL_PRIOR_STD, control_prior_scale=CONTROL_PRIOR_SCALE,
        target_velocity=TARGET_VELOCITY, update_every=UPDATE_EVERY,
        ramp_steps=RAMP_STEPS, forgetting=FORGETTING,
        time_horizon=TIME_HORIZON, debug_first_trial=False)


def aggregate_and_plot(seeds, n_trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from methods import terrain

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    n_slopes_per_seed = {s: terrain.sample_multislope(s)["n_slopes"] for s in seeds}
    print("\n=== terrain per seed ===")
    for s in seeds:
        print(f"  seed {s}: {n_slopes_per_seed[s]} slopes")

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
            if len(best) < n_trials:        # pad short (early-stopped) runs
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
    ax.set_ylim(-1.0, 3.0)   # focus on the converged region; early -50 dips clipped
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best-so-far objective J")
    ax.set_title(f"Multi-slope terrain — best-so-far J, mean ± std over "
                 f"{len(seeds)} seeds (random #slopes per seed)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    out = os.path.join(fig_dir, "comparison_multiseed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="worker mode: one seed")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
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
