"""Online continuous-run comparison on a NATURAL landscape.

One long continuous bout per method on the same natural transect — the robot
walks forward across forward bands of grass / gravel / rocks / river (each band
has its own friction AND its own geometry: gentle undulation, fine roughness,
scattered rocks to step over, and a slippery depression). No resets; each method
updates its CPG parameters at its own cadence:

  * fixed    -> midpoint gait, never adapts (control baseline);
  * grid     -> Latin-hypercube grid search over the first 4.5 s, then held fixed;
  * BO       -> re-optimizes every 4.5 s (TuRBO-style trust region);
  * MARX-EFE -> filters the model every step, re-plans every 1 s.

The natural terrain mixes conflicting per-zone optima (slippery river vs grippy
rocks vs soft grass) — the regime where a single fixed gait may fail but online
adaptation could pay off.

Run from repo root:
    python experiment-natural/run_online.py                 # all seeds + figures
    python experiment-natural/run_online.py --seed 0        # one seed (worker)
    python experiment-natural/run_online.py --seconds 45 --seeds 10
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

    cfg = terrain.sample_natural(seed, reach=seconds * 1.6 + 5.0)
    terrain.TERRAIN_CONFIG = cfg
    print(f"[seed {seed}] natural terrain: {len(cfg['bands'])} bands over "
          f"~{cfg['reach']:.0f} m -> "
          + ", ".join(f"{n}@{y:.1f}" for y, n in cfg["bands"][:8])
          + (" ..." if len(cfg["bands"]) > 8 else ""), flush=True)
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

    save = {"target_v": TARGET_VELOCITY, "reach": cfg["reach"],
            "zones_y": np.array([z[0] for z in cfg["zones"]]),
            "zones_mu": np.array([z[1] for z in cfg["zones"]]),
            "bands_y": np.array([b[0] for b in cfg["bands"]]),
            "bands_name": np.array([b[1] for b in cfg["bands"]]),
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
    from matplotlib.patches import Patch
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
        d = np.load(f, allow_pickle=True); v = float(d["target_v"])
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

    d = np.load(files[0], allow_pickle=True); v = float(d["target_v"])
    ymax = max(d[f"{m}_fy"].max() for m, _ in METHODS if f"{m}_fy" in d)
    by = d["bands_y"]; bn = d["bands_name"]
    edges = list(by) + [ymax + 5]

    def _shade(ax):
        """Tint each forward band by its surface type (natural colours)."""
        for i, y0 in enumerate(by):
            ax.axvspan(y0, edges[i + 1], color=terrain.NATURAL_COLORS[str(bn[i])],
                       alpha=0.18, lw=0)

    band_legend = [Patch(facecolor=terrain.NATURAL_COLORS[n], alpha=0.5,
                   label=f"{n} (μ={terrain.NATURAL_SURFACES[n][0]})")
                   for n in ("grass", "gravel", "rocks", "river")]

    seed0 = re.search(r"seed(\d+)", files[0]).group(1)
    # velocity vs forward position
    fig, ax = plt.subplots(figsize=(12, 5)); _shade(ax)
    for m, c in METHODS:
        ax.plot(d[f"{m}_fy"], d[f"{m}_vx"], color=c, lw=1.1, alpha=0.95, label=METHOD_LABEL[m])
    ax.axhline(v, color="k", ls=":", lw=1, label="target v*")
    ax.set_xlim(0, ymax); ax.set_ylim(-0.2, 2.2)
    ax.set_xlabel("Forward position [m]  (bands tinted by surface type)")
    ax.set_ylabel("Forward velocity vx [m/s]")
    ax.set_title(f"Online velocity tracking across natural terrain (seed {seed0})")
    leg1 = ax.legend(loc="upper right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=band_legend, loc="lower right", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    out = os.path.join(fig_dir, "online_velocity_trace.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"\nsaved {out}")

    # tip-over deviation vs position (recovery)
    figA, axA = plt.subplots(figsize=(12, 5)); _shade(axA)
    for y0 in by:
        if 0 < y0 < ymax:
            axA.axvline(y0, color="gray", ls="--", lw=0.5, alpha=0.35)
    for m, c in METHODS:
        axA.plot(d[f"{m}_fy"], _tip_dev(d[f"{m}_roll"], d[f"{m}_pitch"]),
                 color=c, lw=1.1, alpha=0.95, label=METHOD_LABEL[m])
    axA.set_xlim(0, ymax); axA.set_ylim(0, 90)
    axA.set_xlabel("Forward position [m]  (dashed = band transitions)")
    axA.set_ylabel("Tip deviation  √(roll²+pitch²) [deg]")
    axA.set_title(f"Tip-over stability / recovery at band transitions (seed {seed0})")
    axA.legend(loc="upper right", fontsize=8); axA.grid(True, alpha=0.3)
    outA = os.path.join(fig_dir, "online_attitude_recovery.png")
    figA.savefig(outA, dpi=150, bbox_inches="tight"); print(f"saved {outA}")

    # summary bars: survival distance + fall rate + velocity tracking + tip-over.
    # On natural terrain almost everything eventually falls, so distance-to-fall
    # (survival distance) is the discriminating metric — shown first.
    names = [m for m, _ in METHODS]; labs = [METHOD_LABEL[m] for m in names]
    cols = [c for _, c in METHODS]
    fig2, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(17, 4.2))
    ax0.bar(labs, [np.mean(summ[m]["dist"]) for m in names],
            yerr=[np.std(summ[m]["dist"]) for m in names], color=cols, alpha=0.85, capsize=4)
    ax0.set_ylabel("distance travelled [m]"); ax0.set_title("Survival distance (higher=better)")
    ax1.bar(labs, [100.0 * sum(summ[m]["fell"]) / len(files) for m in names], color=cols, alpha=0.85)
    ax1.set_ylabel("fall rate [%]"); ax1.set_title("Falls (lower=better)")
    ax2.bar(labs, [np.nanmean(summ[m]["err"]) for m in names],
            yerr=[np.nanstd(summ[m]["err"]) for m in names], color=cols, alpha=0.85, capsize=4)
    ax2.set_ylabel("mean |vx - v*| [m/s]"); ax2.set_title("Velocity tracking (lower=better)")
    ax3.bar(labs, [np.nanmean(summ[m]["tip"]) for m in names],
            yerr=[np.nanstd(summ[m]["tip"]) for m in names], color=cols, alpha=0.85, capsize=4)
    ax3.set_ylabel("mean tip dev [deg]"); ax3.set_title("Tip-over (lower=better)")
    for a in (ax0, ax1, ax2, ax3):
        a.grid(True, axis="y", alpha=0.3); a.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig2.suptitle(f"Online continuous run over natural terrain, {seconds:.0f}s, {len(files)} seeds",
                  fontweight="bold")
    fig2.tight_layout()
    out2 = os.path.join(fig_dir, "online_summary.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight"); print(f"saved {out2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        run_one_seed(args.seed, args.seconds)
        return

    seeds = list(range(args.seeds))
    if not args.plot_only:
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
