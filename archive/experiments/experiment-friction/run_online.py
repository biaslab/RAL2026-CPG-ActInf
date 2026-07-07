"""Online continuous-run comparison on varying-friction terrain.

Unlike the episodic experiments (reset every 4.5 s trial), this runs ONE long
continuous bout per method on the same terrain — the robot walks forward across
random friction zones without resets — and each method updates its CPG
parameters at its own cadence, mirroring Zhang et al. (2024):

  * online BO   -> re-optimizes every 4.5 s (one GP update per 4.5 s window,
                   parameters interpolated over the 1.5 s transition);
  * MARX-EFE    -> filters the model every step and re-plans every 1 s.

Grid search is omitted. The question: does MARX-EFE's faster, model-based update
track the target velocity better than BO across friction changes?

Run from repo root:
    python experiment-friction/run_online.py                 # all seeds + figures
    python experiment-friction/run_online.py --seed 0        # one seed (worker)
    python experiment-friction/run_online.py --seconds 45 --seeds 5
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

RESULTS_DIR = os.path.join(_HERE, "results-online")
ROBOT_MASS = 10.0
TARGET_VELOCITY = 1.0
DT = 0.01

# MARX-EFE config (fast solver, 1 s action cadence, buffers 3/10, horizon 2)
CONTROL_PRIOR_SCALE = 0.15
GOAL_PRIOR_STD = (np.sqrt(0.5), np.sqrt(0.5), np.deg2rad(45), np.deg2rad(45))
MARX_UPDATE_EVERY = 100        # 1 s
TIME_HORIZON = 2
# BO config
BO_WINDOW = 450                # 4.5 s
N_INIT = 5
BO_TRUST_RADIUS = 0.2          # TuRBO-style trust region (unit-space half-width)
HUGE = 10 ** 9                 # window_steps > run => no online updates (fixed params)
GRID_K = 20                    # candidates for the grid-over-first-4.5s baseline

# Methods compared (label -> colour). "fixed" and "grid" are non-adaptive.
METHODS = [("fixed", "tab:gray"), ("grid", "tab:green"),
           ("BO", "tab:blue"), ("MARXEFE", "tab:orange")]
METHOD_LABEL = {"fixed": "Fixed (midpoint)", "grid": "Grid@4.5s, then fixed",
                "BO": "BO (4.5 s, TR)", "MARXEFE": "MARX-EFE (1 s)"}


def _grid_initial(mx, robot, jfull, jfilt, jfeet, seed):
    """Grid search (Latin-hypercube) over CPG parameters, each candidate scored on
    a fresh 4.5 s episode (the 'first 4.5 s of data'); returns the best gait by
    velocity tracking. The winner is then held fixed for the whole online bout."""
    from methods.cpg_bounds import bounds
    from scipy.stats.qmc import LatinHypercube
    lo = bounds[0].numpy(); hi = bounds[1].numpy()
    cands = lo + LatinHypercube(d=8, seed=seed).random(GRID_K) * (hi - lo)
    best, bestJ = cands[0], -1e9
    for c in cands:
        log = mx.run_bo_online(robot, jfull, jfilt, jfeet, dt=DT, run_length=4.5,
                               bo=None, target_velocity=TARGET_VELOCITY,
                               robot_mass=ROBOT_MASS, window_steps=HUGE,
                               transition_steps=150, seed=seed, init_params=c)
        t = np.asarray(log["t"]); m = t >= 1.5
        if log["fell"] or m.sum() < 5:
            J = -50.0
        else:
            vx = np.asarray(log["vx"])[m]; vy = np.asarray(log["vy"])[m]
            err = ((vx - TARGET_VELOCITY) ** 2 + vy ** 2) / 0.05
            J = float(np.mean(np.minimum(np.exp(-err), 0.85)))
        if J > bestJ:
            bestJ, best = J, c
    return best


def run_one_seed(seed, seconds):
    from methods import terrain
    import methods.marxefe_optimizer as mx
    from methods.cpg_bounds import bounds
    from methods.bo_optimizer import BOOptimizer, BetaSchedule

    cfg = terrain.sample_friction_long(
        seed, reach=seconds * 1.6 + 5.0, zone_len=(3.0, 6.0),
        palette=("slick", "normal", "grip", "rubber"))
    terrain.TERRAIN_CONFIG = cfg
    print(f"[seed {seed}] friction terrain: {cfg['n_zones']} zones over "
          f"~{cfg['zones'][-1][0]:.0f} m", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mx.load_environment(DT, use_gui=False)
    robot, _, jfull, jfilt, jfeet = mx.load_robot(mx.p, robot_mass=ROBOT_MASS)
    lo = bounds[0].numpy(); hi = bounds[1].numpy(); mid = 0.5 * (lo + hi)

    def keep(L):
        return {k: np.asarray(L[k]) for k in ("t", "vx", "vy", "fy",
                "roll", "pitch", "yaw")} | {"fell": bool(L["fell"])}

    logs = {}

    # 1. Fixed parameters (midpoint gait), no adaptation.
    logs["fixed"] = keep(mx.run_bo_online(
        robot, jfull, jfilt, jfeet, dt=DT, run_length=seconds, bo=None,
        target_velocity=TARGET_VELOCITY, robot_mass=ROBOT_MASS,
        window_steps=HUGE, transition_steps=150, seed=seed, init_params=mid))

    # 2. Grid search over the first 4.5 s, then hold the winner fixed.
    best = _grid_initial(mx, robot, jfull, jfilt, jfeet, seed)
    logs["grid"] = keep(mx.run_bo_online(
        robot, jfull, jfilt, jfeet, dt=DT, run_length=seconds, bo=None,
        target_velocity=TARGET_VELOCITY, robot_mass=ROBOT_MASS,
        window_steps=HUGE, transition_steps=150, seed=seed, init_params=best))

    # 3. Online BO (trust region), update every 4.5 s.
    sched = BetaSchedule(beta_init=5.0, beta_min=1.0, n_decay_start=40, gamma=0.9)
    bo = BOOptimizer(bounds, sched, n_init=N_INIT, seed=seed)
    logs["BO"] = keep(mx.run_bo_online(
        robot, jfull, jfilt, jfeet, dt=DT, run_length=seconds, bo=bo,
        target_velocity=TARGET_VELOCITY, robot_mass=ROBOT_MASS,
        window_steps=BO_WINDOW, transition_steps=150, seed=seed,
        bo_trust_radius=BO_TRUST_RADIUS))

    # 4. MARX-EFE: model every step, re-plan every 1 s.
    agent = mx.build_marx_agent(
        target_velocity=TARGET_VELOCITY, control_prior_scale=CONTROL_PRIOR_SCALE,
        goal_prior_std=GOAL_PRIOR_STD, input_buffer=3, output_buffer=10,
        time_horizon=TIME_HORIZON)
    mx._prev_params_marx = None
    td = mx.run_episode_maxrefe(
        agent, robot, jfull, jfilt, jfeet, dt=DT, episode_length=seconds,
        lambda_energy=1e-2, target_forward_position=4.0,
        update_every=MARX_UPDATE_EVERY, ramp_steps=20)
    logs["MARXEFE"] = keep({"t": td["t"], "vx": td["vx"], "vy": td["vy"],
        "fy": td["base_pos"][:, 1], "roll": td["roll"], "pitch": td["pitch"],
        "yaw": td["yaw"], "fell": td["fall"]})

    save = {"target_v": TARGET_VELOCITY,
            "zones_y": np.array([z[0] for z in cfg["zones"]]),
            "zones_mu": np.array([z[1] for z in cfg["zones"]]),
            "base_mu": cfg["base_mu"]}
    for m, L in logs.items():
        for k in ("t", "vx", "vy", "fy", "roll", "pitch", "yaw"):
            save[f"{m}_{k}"] = L[k]
        save[f"{m}_fell"] = L["fell"]
    np.savez(os.path.join(RESULTS_DIR, f"online_seed{seed}.npz"), **save)
    print("[seed %d] " % seed + " | ".join(
        f"{m} fell={logs[m]['fell']} dist={logs[m]['fy'][-1]:.1f}m"
        for m, _ in METHODS), flush=True)


def _track_err(vx, v_star, t, settle=1.5):
    m = np.asarray(t) >= settle
    return float(np.mean(np.abs(np.asarray(vx)[m] - v_star))) if m.any() else np.nan


def _tip_dev(roll, pitch):
    """Tip-over deviation √(roll²+pitch²) in degrees (spikes at perturbations and
    recovers — the cleaner stability/recovery signal)."""
    return np.rad2deg(np.sqrt(np.asarray(roll) ** 2 + np.asarray(pitch) ** 2))


def _yaw_drift(yaw):
    """Heading deviation |yaw| in degrees (accumulates; not a 'recovery' signal)."""
    return np.rad2deg(np.abs(np.asarray(yaw)))


def _mean_after(series, t, settle=1.5):
    m = np.asarray(t) >= settle
    return float(np.mean(np.asarray(series)[m])) if m.any() else np.nan


def aggregate_and_plot(seeds, seconds):
    import glob, re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from methods import terrain

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "online_seed*.npz")),
                   key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
    if not files:
        raise SystemExit("no online results found")

    print("\n=== online tracking / stability / falls, over seeds (4 methods) ===")
    summ = {m: {"err": [], "tip": [], "yaw": [], "fell": [], "dist": []} for m, _ in METHODS}
    for f in files:
        d = np.load(f); v = float(d["target_v"])
        for m, _ in METHODS:
            if f"{m}_vx" not in d:
                continue
            summ[m]["err"].append(_track_err(d[f"{m}_vx"], v, d[f"{m}_t"]))
            summ[m]["tip"].append(_mean_after(_tip_dev(d[f"{m}_roll"], d[f"{m}_pitch"]), d[f"{m}_t"]))
            summ[m]["yaw"].append(_mean_after(_yaw_drift(d[f"{m}_yaw"]), d[f"{m}_t"]))
            summ[m]["fell"].append(bool(d[f"{m}_fell"]))
            summ[m]["dist"].append(float(d[f"{m}_fy"][-1]))
    for m, _ in METHODS:
        e = np.array(summ[m]["err"]); tp = np.array(summ[m]["tip"]); yw = np.array(summ[m]["yaw"])
        print(f"  {METHOD_LABEL[m]:22s} err={np.nanmean(e):.3f}±{np.nanstd(e):.3f} | "
              f"tip={np.nanmean(tp):.1f}±{np.nanstd(tp):.1f}deg | yaw={np.nanmean(yw):.1f}deg | "
              f"falls={sum(summ[m]['fell'])}/{len(files)} | dist={np.mean(summ[m]['dist']):.1f}m")

    d = np.load(files[0]); v = float(d["target_v"])
    ymax = max(d[f"{m}_fy"].max() for m, _ in METHODS if f"{m}_fy" in d)
    zy, zmu = d["zones_y"], d["zones_mu"]
    edges = list(zy) + [ymax + 5]

    def _shade(ax):
        for i, y0 in enumerate(zy):
            if zmu[i] < 0.5:
                ax.axvspan(y0, edges[i + 1], color="tab:blue",
                           alpha=0.10 + 0.12 * (0.5 - zmu[i]) / 0.5, lw=0)

    seed0 = re.search(r"seed(\d+)", files[0]).group(1)
    # velocity vs forward position
    fig, ax = plt.subplots(figsize=(12, 5)); _shade(ax)
    for m, c in METHODS:
        ax.plot(d[f"{m}_fy"], d[f"{m}_vx"], color=c, lw=1.0, alpha=0.9, label=METHOD_LABEL[m])
    ax.axhline(v, color="k", ls=":", lw=1, label="target v*")
    ax.set_xlim(0, ymax); ax.set_ylim(-0.2, 2.2)
    ax.set_xlabel("Forward position [m]  (shaded = low-friction zones)")
    ax.set_ylabel("Forward velocity vx [m/s]")
    ax.set_title(f"Online velocity tracking across friction zones (seed {seed0})")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3)
    out = os.path.join(fig_dir, "online_velocity_trace.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"\nsaved {out}")

    # tip-over deviation vs position (recovery)
    figA, axA = plt.subplots(figsize=(12, 5)); _shade(axA)
    for y0 in zy:
        if 0 < y0 < ymax:
            axA.axvline(y0, color="gray", ls="--", lw=0.5, alpha=0.35)
    for m, c in METHODS:
        axA.plot(d[f"{m}_fy"], _tip_dev(d[f"{m}_roll"], d[f"{m}_pitch"]),
                 color=c, lw=1.0, alpha=0.9, label=METHOD_LABEL[m])
    axA.set_xlim(0, ymax); axA.set_ylim(0, 90)
    axA.set_xlabel("Forward position [m]  (dashed = transitions; shaded = low-friction)")
    axA.set_ylabel("Tip deviation  √(roll²+pitch²) [deg]")
    axA.set_title(f"Tip-over stability / recovery at friction transitions (seed {seed0})")
    axA.legend(loc="upper right", fontsize=8); axA.grid(True, alpha=0.3)
    outA = os.path.join(fig_dir, "online_attitude_recovery.png")
    figA.savefig(outA, dpi=150, bbox_inches="tight"); print(f"saved {outA}")

    # summary bars: fall rate + velocity tracking + tip-over stability
    names = [m for m, _ in METHODS]; labs = [METHOD_LABEL[m] for m in names]
    cols = [c for _, c in METHODS]
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    ax1.bar(labs, [100.0 * sum(summ[m]["fell"]) / len(files) for m in names], color=cols, alpha=0.85)
    ax1.set_ylabel("fall rate [%]"); ax1.set_title("Falls (lower=better)")
    ax2.bar(labs, [np.nanmean(summ[m]["err"]) for m in names],
            yerr=[np.nanstd(summ[m]["err"]) for m in names], color=cols, alpha=0.85, capsize=4)
    ax2.set_ylabel("mean |vx - v*| [m/s]"); ax2.set_title("Velocity tracking (lower=better)")
    ax3.bar(labs, [np.nanmean(summ[m]["tip"]) for m in names],
            yerr=[np.nanstd(summ[m]["tip"]) for m in names], color=cols, alpha=0.85, capsize=4)
    ax3.set_ylabel("mean tip dev [deg]"); ax3.set_title("Tip-over (lower=better)")
    for a in (ax1, ax2, ax3):
        a.grid(True, axis="y", alpha=0.3); a.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig2.suptitle(f"Online continuous run over {seconds:.0f}s, {len(files)} seeds",
                  fontweight="bold")
    fig2.tight_layout()
    out2 = os.path.join(fig_dir, "online_summary.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight"); print(f"saved {out2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    if args.seed is not None:
        run_one_seed(args.seed, args.seconds)
        return

    seeds = list(range(args.seeds))
    t0 = time.time()
    for s in seeds:
        print(f"\n########## SEED {s} / {seeds[-1]} ##########", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--seed", str(s), "--seconds", str(args.seconds)],
                       check=True, cwd=_REPO_ROOT)
    print(f"\nAll seeds done in {(time.time()-t0)/60:.1f} min.")
    aggregate_and_plot(seeds, args.seconds)


if __name__ == "__main__":
    main()
