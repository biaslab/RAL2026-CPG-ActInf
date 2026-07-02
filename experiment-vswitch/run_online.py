"""Non-stationary velocity-switch comparison on a discrete-obstacle field.

The commanded forward velocity v* changes partway through a single continuous
run (no resets), while the robot walks over Isaac-Gym-style scattered obstacles.
This is the regime where online adaptation *should* beat a single fixed gait: no
one gait is optimal for every commanded speed.

To keep the comparison fair, every method re-optimizes per velocity segment:

  * fixed    -> gait grid-searched for the FIRST target, then held all run
                (the "tune once, never re-tune" baseline);
  * grid     -> gait grid-searched SEPARATELY for each target velocity
                (re-run grid search after every switch), held within a segment;
  * BO       -> online BO; the GP is rebuilt fresh at each switch so it
                re-optimizes from scratch for the new target (BO analogue of
                re-running grid), updating every 4.5 s;
  * MARX-EFE -> the velocity goal is switched at each transition; the learned
                dynamics model is kept and it re-plans every 1 s.

Run from repo root:
    python experiment-velocity-switch/run_online.py                  # all seeds
    python experiment-velocity-switch/run_online.py --seed 0         # one worker
    python experiment-velocity-switch/run_online.py --seeds 8 --seg 10
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
DT = 0.01

# Velocity schedule: target forward speed per segment (m/s). Switches are evenly
# spaced over the run; the schedule defines both the task and #segments.
VELOCITIES = [1.0, 1.6, 0.7, 1.3]
SETTLE_S = 1.5                  # ignore this long after each switch in the metric
# Obstacle field: the open-loop CPG gait cannot sustain traversal of a discrete
# obstacle field (falls within seconds), so the *measured* velocity-switch run is
# on flat ground (density 0) to keep tracking the clean discriminator. Set
# OBSTACLE_DENSITY > 0 (e.g. 0.3) to add smooth mounds — expect ~100% falls.
OBSTACLE_DENSITY = 0.0
OBSTACLE_HEIGHT = (0.04, 0.07)

# MARX-EFE config
CONTROL_PRIOR_SCALE = 0.15
GOAL_PRIOR_STD = (np.sqrt(0.5), np.sqrt(0.5), np.deg2rad(45), np.deg2rad(45))
MARX_UPDATE_EVERY = 100        # 1 s
TIME_HORIZON = 2
# BO config
BO_WINDOW = 450                # 4.5 s
N_INIT = 5
BO_TRUST_RADIUS = 0.2
HUGE = 10 ** 9
GRID_K = 12                    # candidates per velocity for the grid baselines

METHODS = [("fixed", "tab:gray"), ("grid", "tab:green"),
           ("BO", "tab:blue"), ("MARXEFE", "tab:orange")]
METHOD_LABEL = {"fixed": "Fixed (tuned @v0)", "grid": "Grid (re-tuned/segment)",
                "BO": "BO (restart/segment)", "MARXEFE": "MARX-EFE (1 s)"}


def _grid_for_velocity(mx, robot, jfull, jfilt, jfeet, seed, v):
    """Grid search (Latin-hypercube, GRID_K candidates) for a single target
    velocity `v`; each candidate scored on a fresh 4.5 s episode. Returns the
    best gait. Called once per distinct target velocity ('re-run grid')."""
    from methods.cpg_bounds import bounds
    from scipy.stats.qmc import LatinHypercube
    lo = bounds[0].numpy(); hi = bounds[1].numpy()
    cands = lo + LatinHypercube(d=8, seed=seed * 17 + int(v * 10)).random(GRID_K) * (hi - lo)
    best, bestJ = cands[0], -1e9
    for c in cands:
        log = mx.run_bo_online(robot, jfull, jfilt, jfeet, dt=DT, run_length=4.5,
                               bo=None, target_velocity=v, robot_mass=ROBOT_MASS,
                               window_steps=HUGE, transition_steps=150,
                               seed=seed, init_params=c)
        t = np.asarray(log["t"]); m = t >= 1.5
        if log["fell"] or m.sum() < 5:
            J = -50.0
        else:
            vx = np.asarray(log["vx"])[m]; vy = np.asarray(log["vy"])[m]
            err = ((vx - v) ** 2 + vy ** 2) / 0.05
            J = float(np.mean(np.minimum(np.exp(-err), 0.85)))
        if J > bestJ:
            bestJ, best = J, c
    return best


def run_one_seed(seed, seg_seconds):
    from methods import terrain
    import methods.marxefe_optimizer as mx
    from methods.cpg_bounds import bounds
    from methods.bo_optimizer import BOOptimizer, BetaSchedule

    vels = list(VELOCITIES)
    n_seg = len(vels)
    seconds = seg_seconds * n_seg
    seg_steps = int(seg_seconds / DT)
    num_steps = int(seconds / DT)
    # forward distance needed ~ sum(v)*seg + margin
    reach = float(sum(vels) * seg_seconds + 8.0)

    cfg = terrain.sample_obstacles(seed, reach=reach, density=OBSTACLE_DENSITY,
                                   height=OBSTACLE_HEIGHT, size=(0.4, 0.8))
    terrain.TERRAIN_CONFIG = cfg
    print(f"[seed {seed}] terrain: {cfg['n_obs']} obstacles over ~{reach:.0f} m "
          f"({'flat' if cfg['n_obs'] == 0 else 'mounds'}) | "
          f"v* schedule {vels} every {seg_seconds:.0f}s", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mx.load_environment(DT, use_gui=False)
    robot, _, jfull, jfilt, jfeet = mx.load_robot(mx.p, robot_mass=ROBOT_MASS)

    # Per-step target-velocity schedule.
    tsched = np.empty(num_steps)
    for i in range(n_seg):
        tsched[i * seg_steps:(i + 1) * seg_steps] = vels[i]
    tsched[(n_seg) * seg_steps:] = vels[-1]

    # Pre-compute the grid optimum for each distinct target velocity ('re-run
    # grid search after each switch' — done up front so the main bouts stay
    # continuous / reset-free).
    gait_for = {}
    for v in dict.fromkeys(vels):
        gait_for[v] = _grid_for_velocity(mx, robot, jfull, jfilt, jfeet, seed, v)
    print(f"[seed {seed}] grid optima ready for {list(gait_for)}", flush=True)

    switch_steps = [i * seg_steps for i in range(1, n_seg)]
    grid_schedule = [(i * seg_steps, gait_for[vels[i]]) for i in range(1, n_seg)]

    def keep(L):
        return {k: np.asarray(L[k]) for k in ("t", "vx", "vy", "fy",
                "roll", "pitch", "yaw", "target")} | {"fell": bool(L["fell"])}

    logs = {}

    # 1. Fixed: grid optimum for the first target, held all run.
    logs["fixed"] = keep(mx.run_bo_online(
        robot, jfull, jfilt, jfeet, dt=DT, run_length=seconds, bo=None,
        target_velocity=vels[0], robot_mass=ROBOT_MASS, window_steps=HUGE,
        transition_steps=150, seed=seed, init_params=gait_for[vels[0]],
        target_schedule=tsched))

    # 2. Grid: re-tuned per segment (switch to that segment's grid optimum).
    logs["grid"] = keep(mx.run_bo_online(
        robot, jfull, jfilt, jfeet, dt=DT, run_length=seconds, bo=None,
        target_velocity=vels[0], robot_mass=ROBOT_MASS, window_steps=HUGE,
        transition_steps=150, seed=seed, init_params=gait_for[vels[0]],
        target_schedule=tsched, param_schedule=grid_schedule))

    # 3. Online BO: rebuild the GP at each switch, update every 4.5 s.
    def bo_factory():
        sched = BetaSchedule(beta_init=5.0, beta_min=1.0, n_decay_start=40, gamma=0.9)
        return BOOptimizer(bounds, sched, n_init=N_INIT, seed=seed)
    logs["BO"] = keep(mx.run_bo_online(
        robot, jfull, jfilt, jfeet, dt=DT, run_length=seconds, bo=bo_factory(),
        target_velocity=vels[0], robot_mass=ROBOT_MASS, window_steps=BO_WINDOW,
        transition_steps=150, seed=seed, bo_trust_radius=BO_TRUST_RADIUS,
        init_params=gait_for[vels[0]], target_schedule=tsched,
        bo_factory=bo_factory, restart_steps=switch_steps))

    # 4. MARX-EFE: switch the velocity goal at each transition; model is kept.
    agent = mx.build_marx_agent(
        target_velocity=vels[0], control_prior_scale=CONTROL_PRIOR_SCALE,
        goal_prior_std=GOAL_PRIOR_STD, input_buffer=3, output_buffer=10,
        time_horizon=TIME_HORIZON)
    mx._prev_params_marx = None
    td = mx.run_episode_maxrefe(
        agent, robot, jfull, jfilt, jfeet, dt=DT, episode_length=seconds,
        lambda_energy=1e-2, target_forward_position=4.0,
        update_every=MARX_UPDATE_EVERY, ramp_steps=20, target_schedule=tsched)
    logs["MARXEFE"] = keep({"t": td["t"], "vx": td["vx"], "vy": td["vy"],
        "fy": td["base_pos"][:, 1], "roll": td["roll"], "pitch": td["pitch"],
        "yaw": td["yaw"], "target": td["target"], "fell": td["fall"]})

    save = {"velocities": np.array(vels), "seg_seconds": seg_seconds,
            "seconds": seconds, "reach": reach}
    for m, L in logs.items():
        for k in ("t", "vx", "vy", "fy", "roll", "pitch", "yaw", "target"):
            save[f"{m}_{k}"] = L[k]
        save[f"{m}_fell"] = L["fell"]
    np.savez(os.path.join(RESULTS_DIR, f"vswitch_seed{seed}.npz"), **save)
    print("[seed %d] " % seed + " | ".join(
        f"{m} fell={logs[m]['fell']} err={_track_err(logs[m]['vx'], logs[m]['target'], logs[m]['t'], vels, seg_seconds):.3f}"
        for m, _ in METHODS), flush=True)


def _settle_mask(t, vels, seg_seconds, settle=SETTLE_S):
    """Boolean mask over steps that are >= `settle` s into their segment (i.e.
    exclude the post-switch transient) — where steady tracking is measured."""
    t = np.asarray(t)
    within = t - (np.floor(t / seg_seconds) * seg_seconds)
    return within >= settle


def _track_err(vx, target, t, vels, seg_seconds, settle=SETTLE_S):
    m = _settle_mask(t, vels, seg_seconds, settle)
    if not m.any():
        return np.nan
    return float(np.mean(np.abs(np.asarray(vx)[m] - np.asarray(target)[m])))


def _tip_dev(roll, pitch):
    return np.rad2deg(np.sqrt(np.asarray(roll) ** 2 + np.asarray(pitch) ** 2))


def _mean_after(series, t, vels, seg_seconds, settle=SETTLE_S):
    m = _settle_mask(t, vels, seg_seconds, settle)
    return float(np.mean(np.asarray(series)[m])) if m.any() else np.nan


def aggregate_and_plot(seeds, seg_seconds):
    import glob, re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "vswitch_seed*.npz")),
                   key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
    if not files:
        raise SystemExit("no velocity-switch results found")

    vels = list(np.load(files[0])["velocities"])
    print("\n=== velocity-switch tracking / stability / falls (4 methods) ===")
    summ = {m: {"err": [], "tip": [], "fell": []} for m, _ in METHODS}
    perseg = {m: [] for m, _ in METHODS}
    for f in files:
        d = np.load(f)
        for m, _ in METHODS:
            if f"{m}_vx" not in d:
                continue
            t = d[f"{m}_t"]; vx = d[f"{m}_vx"]; tg = d[f"{m}_target"]
            summ[m]["err"].append(_track_err(vx, tg, t, vels, seg_seconds))
            summ[m]["tip"].append(_mean_after(_tip_dev(d[f"{m}_roll"], d[f"{m}_pitch"]), t, vels, seg_seconds))
            summ[m]["fell"].append(bool(d[f"{m}_fell"]))
            # per-segment steady error (only segments the run reached)
            segerr = []
            for i, v in enumerate(vels):
                sm = (t >= i * seg_seconds + SETTLE_S) & (t < (i + 1) * seg_seconds)
                segerr.append(float(np.mean(np.abs(vx[sm] - v))) if sm.any() else np.nan)
            perseg[m].append(segerr)
    for m, _ in METHODS:
        e = np.array(summ[m]["err"]); tp = np.array(summ[m]["tip"])
        print(f"  {METHOD_LABEL[m]:24s} err={np.nanmean(e):.3f}±{np.nanstd(e):.3f} | "
              f"tip={np.nanmean(tp):.1f}deg | falls={sum(summ[m]['fell'])}/{len(files)}")

    # velocity trace vs TIME (seed 0) with the step-target overlaid
    d = np.load(files[0]); seed0 = re.search(r"seed(\d+)", files[0]).group(1)
    n_seg = len(vels)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(n_seg):
        if i % 2 == 1:
            ax.axvspan(i * seg_seconds, (i + 1) * seg_seconds, color="k", alpha=0.04, lw=0)
        ax.axvline(i * seg_seconds, color="gray", ls="--", lw=0.6, alpha=0.5)
    # target step
    tt = np.arange(0, n_seg * seg_seconds, DT)
    tg = np.array([vels[min(int(x / seg_seconds), n_seg - 1)] for x in tt])
    ax.step(tt, tg, color="k", ls=":", lw=1.5, where="post", label="target v*")
    for m, c in METHODS:
        ax.plot(d[f"{m}_t"], d[f"{m}_vx"], color=c, lw=1.1, alpha=0.95, label=METHOD_LABEL[m])
    ax.set_xlim(0, n_seg * seg_seconds); ax.set_ylim(-0.2, 2.3)
    ax.set_xlabel("Time [s]  (dashed = velocity switches)")
    ax.set_ylabel("Forward velocity vx [m/s]")
    ax.set_title(f"Velocity-switch tracking over obstacles (seed {seed0})")
    ax.legend(loc="upper right", fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    out = os.path.join(fig_dir, "vswitch_velocity_trace.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"\nsaved {out}")

    # summary: tracking err + falls + tip + per-segment err
    names = [m for m, _ in METHODS]; labs = [METHOD_LABEL[m] for m in names]
    cols = [c for _, c in METHODS]
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4.3))
    ax1.bar(labs, [np.nanmean(summ[m]["err"]) for m in names],
            yerr=[np.nanstd(summ[m]["err"]) for m in names], color=cols, alpha=0.85, capsize=4)
    ax1.set_ylabel("mean |vx - v*(t)| [m/s]"); ax1.set_title("Velocity tracking (lower=better)")
    ax2.bar(labs, [100.0 * sum(summ[m]["fell"]) / len(files) for m in names], color=cols, alpha=0.85)
    ax2.set_ylabel("fall rate [%]"); ax2.set_title("Falls (lower=better)")
    ax3.bar(labs, [np.nanmean(summ[m]["tip"]) for m in names],
            yerr=[np.nanstd(summ[m]["tip"]) for m in names], color=cols, alpha=0.85, capsize=4)
    ax3.set_ylabel("mean tip dev [deg]"); ax3.set_title("Tip-over (lower=better)")
    x = np.arange(n_seg); w = 0.2
    for j, m in enumerate(names):
        arr = np.array(perseg[m], dtype=float)
        ax4.bar(x + (j - 1.5) * w, np.nanmean(arr, axis=0), w, color=cols[j], alpha=0.85,
                label=METHOD_LABEL[m])
    ax4.set_xticks(x); ax4.set_xticklabels([f"v*={v}" for v in vels], fontsize=8)
    ax4.set_ylabel("steady err [m/s]"); ax4.set_title("Per-segment tracking")
    ax4.legend(fontsize=6)
    for a in (ax1, ax2, ax3):
        a.grid(True, axis="y", alpha=0.3); a.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax4.grid(True, axis="y", alpha=0.3)
    fig2.suptitle(f"Non-stationary velocity switching over obstacles, "
                  f"{n_seg}x{seg_seconds:.0f}s, {len(files)} seeds", fontweight="bold")
    fig2.tight_layout()
    out2 = os.path.join(fig_dir, "vswitch_summary.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight"); print(f"saved {out2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seg", type=float, default=10.0, help="seconds per velocity segment")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--obs", type=float, default=None,
                    help="obstacle density (mounds/m); 0=flat. Default flat.")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.obs is not None:
        global OBSTACLE_DENSITY
        OBSTACLE_DENSITY = args.obs

    if args.seed is not None:
        run_one_seed(args.seed, args.seg)
        return

    seeds = list(range(args.seeds))
    if not args.plot_only:
        t0 = time.time()
        for s in seeds:
            print(f"\n########## SEED {s} / {seeds[-1]} ##########", flush=True)
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--seed", str(s), "--seg", str(args.seg),
                            "--obs", str(OBSTACLE_DENSITY)],
                           check=True, cwd=_REPO_ROOT)
        print(f"\nAll seeds done in {(time.time()-t0)/60:.1f} min.")
    aggregate_and_plot(seeds, args.seg)


if __name__ == "__main__":
    main()
