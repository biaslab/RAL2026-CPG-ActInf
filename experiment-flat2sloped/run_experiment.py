"""Flat-to-sloped terrain switch: keep flat-optimal vs switch to sloped-optimal.

Hypothesis test: if the global optimum of the CPG parameters shifts with
terrain, then switching to the sloped-terrain optimum when the terrain becomes
sloped should outperform keeping the flat-terrain optimum.

Design
------
* The robot walks with the flat-optimal parameters on flat ground for 10 s.
  A per-seed calibration run (identical dynamics, slope pushed out of reach)
  measures the base's forward position y10 at t = 10 s; the main runs then
  place the ramp (10 deg) so it starts exactly at y10 — the terrain "shifts
  to sloped" at t = 10 s.
* Two paired conditions per seed, identical up to the switch step:
    keep   — hold the flat-optimal parameters throughout,
    switch — ramp to the sloped-optimal parameters over 0.4 s at t = 10 s.
* No randomization over starting points: every run starts from the same
  nominal pose at the origin. Seeds differ only through a tiny initial
  joint-angle jitter (sigma = 0.002 rad), which the chaotic contact dynamics
  amplify; this gives independent realizations of the same experiment.
* Metrics are computed over the post-switch window (t in [10, 20] s):
  fall, forward distance, mean forward velocity, RMS roll/pitch, and the
  shared optimization objective J = velocity-tracking reward (v* = 0.5 m/s)
  minus 0.5 x cost of transport (Zhang et al. eq. 9, as in the BO pipeline).

Parameter vectors come from figures/cpg_optima_by_parameter.csv (5 BO seeds
per terrain). `preselect` evaluates all 5 candidates per terrain on their own
terrain (8 jitter repetitions) and picks the best one by (fewest falls, then
highest mean J); the fastest-climbing sloped candidate is kept as a secondary
"switch_fast" arm when it differs. `run` executes the 100-seed main
experiment; `aggregate` produces the figure and paired statistics.

Usage (from repo root):
    python experiment-flat2sloped/run_experiment.py preselect
    python experiment-flat2sloped/run_experiment.py run [--seeds 100 --workers 10]
    python experiment-flat2sloped/run_experiment.py aggregate
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_HERE, "results")
OPTIMA_CSV = os.path.join(_REPO, "figures", "cpg_optima_by_parameter.csv")
SELECTED_JSON = os.path.join(RESULTS_DIR, "selected_params.json")
MAIN_CSV = os.path.join(RESULTS_DIR, "switch_experiment.csv")

# The episode runner, jitter reset, and shared constants live in methods.episode
# (single source of truth for the attitude convention and dynamics).
from methods.episode import (DT, SWITCH_RAMP_STEPS, SLOPE_DEG, N_COLS,       # noqa: E402
                             FAR_SLOPE_Y, JITTER_STD, TARGET_VX, ROBOT_MASS, G,
                             DEFAULT_ORI, LEG_NAMES, PARAM_NAMES,
                             _reset_with_jitter, run_episode)

T_SWITCH = 10.0           # terrain + parameter switch time [s]
T_TOTAL = 20.0            # episode length after settling [s]


def load_optima():
    """Candidate optima per terrain from figures/cpg_optima_by_parameter.csv."""
    out = {}
    with open(OPTIMA_CSV) as f:
        for row in csv.DictReader(f):
            vec = np.array([float(row[k]) for k in
                            ["coupling_gain", "w_swing", "w_stance", "F_FAST",
                             "STOP_GAIN", "hip_amp", "knee_amp", "b"]])
            out.setdefault(row["terrain"], []).append((int(row["seed"]), vec))
    return out




def window_metrics(log, k0, k1):
    """Metrics over control steps [k0, k1): distance, velocity, stability,
    tracking reward, CoT and full objective J (as in bo_optimizer.compute_
    objective, with a fall scored J = -50), and whether the robot fell."""
    n = len(log["y"])
    fell_in = bool(log["fell"]) and log["fall_step"] >= k0
    k_end = min(k1, n)
    if k_end <= k0:          # fell before the window opened
        return dict(fell=True, t_fall=(log["fall_step"] * DT if log["fell"] else np.nan),
                    dist=0.0, mean_vx=0.0, rms_roll=np.nan, rms_pitch=np.nan,
                    J_track=0.0, CoT=np.nan, J=-50.0)
    y = log["y"][k0:k_end]
    vx = log["vx"][k0:k_end]
    roll = log["roll"][k0:k_end]
    pitch = log["pitch"][k0:k_end]
    err = (vx - TARGET_VX) ** 2 / 0.05
    J_track = float(DT * np.sum(np.minimum(np.exp(-err), 0.85)))
    dist = float(y[-1] - (log["y"][k0 - 1] if k0 > 0 else 0.0))
    mech = float(DT * np.sum(log["power"][k0:k_end]))
    cap = 200.0 if dist < 0.5 else (150.0 if dist < 1.5 else 100.0)
    CoT = min(mech / (ROBOT_MASS * G * max(abs(dist), 0.001)), cap)
    J = -50.0 if fell_in else J_track - 0.5 * CoT
    return dict(
        fell=fell_in,
        t_fall=(log["fall_step"] * DT if fell_in else np.nan),
        dist=dist,
        mean_vx=float(np.mean(vx)),
        rms_roll=float(np.rad2deg(np.sqrt(np.mean(roll ** 2)))),
        rms_pitch=float(np.rad2deg(np.sqrt(np.mean(pitch ** 2)))),
        J_track=J_track,
        CoT=CoT,
        J=J,
    )


# ── Stage 1: candidate pre-selection ─────────────────────────────────────────

def _preselect_job(job):
    terrain_name, cand_seed, params, rep = job
    if terrain_name == "flat":
        cfg = {"kind": "sloped", "slope_deg": SLOPE_DEG,
               "slope_start_y": FAR_SLOPE_Y, "n_cols": N_COLS}
    else:  # sloped: ramp from 2 m on, robot climbs for most of the episode
        cfg = {"kind": "sloped", "slope_deg": SLOPE_DEG,
               "slope_start_y": 2.0, "n_cols": N_COLS}
    log = run_episode(cfg, seed=1000 + rep, params_start=params, duration=10.0)
    m = window_metrics(log, 0, int(10.0 / DT))
    m["fell"] = bool(log["fell"])   # any fall in the 10 s episode counts
    return terrain_name, cand_seed, rep, m


def preselect(workers):
    optima = load_optima()
    jobs = [(tname, cs, vec, rep)
            for tname in ("flat", "sloped")
            for (cs, vec) in optima[tname]
            for rep in range(8)]
    ctx = get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        results = pool.map(_preselect_job, jobs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    selected = {}
    for tname in ("flat", "sloped"):
        print(f"\n=== {tname} candidates (8 reps each, 10 s on own terrain) ===")
        scores, dist_scores = [], []
        for cs, vec in optima[tname]:
            ms = [m for (tn, c, _, m) in results if tn == tname and c == cs]
            falls = sum(m["fell"] for m in ms)
            dist = np.mean([m["dist"] for m in ms])
            J = np.mean([m["J"] for m in ms])
            print(f"  BO seed {cs}: falls {falls}/8, mean dist {dist:5.2f} m, "
                  f"mean J {J:6.2f}")
            scores.append((falls, -J, cs, vec))
            dist_scores.append((falls, -dist, cs, vec))
        falls, negJ, cs, vec = sorted(scores)[0]
        print(f"  -> selected BO seed {cs} ({falls}/8 falls, J = {-negJ:.2f})")
        selected[tname] = {"bo_seed": cs, "params": vec.tolist()}
        if tname == "sloped":
            f2, negd, cs2, vec2 = sorted(dist_scores)[0]
            if cs2 != cs:
                print(f"  -> secondary (fastest climber): BO seed {cs2} "
                      f"({f2}/8 falls, {-negd:.2f} m)")
                selected["sloped_fast"] = {"bo_seed": cs2, "params": vec2.tolist()}

    with open(SELECTED_JSON, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nsaved {SELECTED_JSON}")


# ── Stage 2: main 100-seed switch experiment ─────────────────────────────────

def _seed_job(job):
    seed, arms = job          # arms: {name: 8-vector or None (= keep)}
    params_flat = np.asarray(arms["keep"])
    k_switch = int(round(T_SWITCH / DT))

    # Calibration: same dynamics, slope out of reach; find y at t = 10 s.
    cal_cfg = {"kind": "sloped", "slope_deg": SLOPE_DEG,
               "slope_start_y": FAR_SLOPE_Y, "n_cols": N_COLS}
    cal = run_episode(cal_cfg, seed, params_flat, duration=T_SWITCH)
    if cal["fell"]:
        return dict(seed=seed, valid=0, reason="fell_calibration",
                    y10=np.nan)
    y10 = float(cal["y"][-1])

    cfg = {"kind": "sloped", "slope_deg": SLOPE_DEG,
           "slope_start_y": y10, "n_cols": N_COLS}
    logs = {}
    for name, target in arms.items():
        if name == "keep":
            logs[name] = run_episode(cfg, seed, params_flat, duration=T_TOTAL)
        else:
            logs[name] = run_episode(cfg, seed, params_flat,
                                     params_target=np.asarray(target),
                                     switch_step=k_switch, duration=T_TOTAL)

    # All conditions are identical before the switch; a pre-switch fall
    # invalidates the seed.
    keep = logs["keep"]
    if keep["fell"] and keep["fall_step"] < k_switch:
        return dict(seed=seed, valid=0, reason="fell_pre_switch", y10=y10)

    # Paired-prefix sanity check.
    n_pre = min([k_switch] + [len(lg["y"]) for lg in logs.values()])
    prefix_gap = float(max(np.max(np.abs(keep["y"][:n_pre] - lg["y"][:n_pre]))
                           for lg in logs.values()))

    row = dict(seed=seed, valid=1, reason="", y10=y10, prefix_gap=prefix_gap)
    for name, log in logs.items():
        m = window_metrics(log, k_switch, int(T_TOTAL / DT))
        for k, v in m.items():
            row[f"{name}_{k}"] = v
    return row


def run_main(seeds, workers):
    with open(SELECTED_JSON) as f:
        sel = json.load(f)
    arms = {"keep": sel["flat"]["params"], "switch": sel["sloped"]["params"]}
    print("keep   (flat-optimal)  :", np.round(sel["flat"]["params"], 3).tolist(),
          f"(BO seed {sel['flat']['bo_seed']})")
    print("switch (sloped-optimal):", np.round(sel["sloped"]["params"], 3).tolist(),
          f"(BO seed {sel['sloped']['bo_seed']})")
    if "sloped_fast" in sel:
        arms["switch_fast"] = sel["sloped_fast"]["params"]
        print("switch_fast (fastest climber):",
              np.round(sel["sloped_fast"]["params"], 3).tolist(),
              f"(BO seed {sel['sloped_fast']['bo_seed']})")
    arm_names = [a for a in arms if a != "keep"]

    jobs = [(s, arms) for s in range(seeds)]
    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        for i, row in enumerate(pool.imap_unordered(_seed_job, jobs)):
            rows.append(row)
            if row["valid"]:
                msg = (f"[{i+1:3d}/{seeds}] seed {row['seed']:3d}  "
                       f"y10={row['y10']:5.2f} m  "
                       f"keep: {'FELL' if row['keep_fell'] else '  ok'} "
                       f"d={row['keep_dist']:5.2f} J={row['keep_J']:6.2f}")
                for a in arm_names:
                    msg += (f"   {a}: {'FELL' if row[f'{a}_fell'] else '  ok'} "
                            f"d={row[f'{a}_dist']:5.2f} J={row[f'{a}_J']:6.2f}")
                print(msg, flush=True)
            else:
                print(f"[{i+1:3d}/{seeds}] seed {row['seed']:3d}  INVALID ({row['reason']})",
                      flush=True)

    rows.sort(key=lambda r: r["seed"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "seed", k))
    with open(MAIN_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {MAIN_CSV}")


# ── Stage 3: aggregation, statistics, figure ─────────────────────────────────

def aggregate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(MAIN_CSV)
    n_total = len(df)
    df = df[df["valid"] == 1].copy()
    n = len(df)
    arms = [c[:-len("_dist")] for c in df.columns
            if c.endswith("_dist") and c != "keep_dist"]
    print(f"{n}/{n_total} seeds valid (pre-switch falls excluded)")
    print(f"max paired-prefix gap: {df['prefix_gap'].max():.2e} m")
    print(f"y10 (slope start): {df['y10'].mean():.2f} +/- {df['y10'].std():.2f} m "
          f"[{df['y10'].min():.2f}, {df['y10'].max():.2f}]")

    kf = df["keep_fell"].astype(bool)
    for arm in arms:
        af = df[f"{arm}_fell"].astype(bool)
        only_keep, only_arm = int((kf & ~af).sum()), int((~kf & af).sum())
        print(f"\n── {arm} vs keep ──────────────────────────────────────────")
        print(f"falls on slope (10 s post-switch): keep {kf.sum()}/{n} "
              f"({100*kf.mean():.0f} %), {arm} {af.sum()}/{n} ({100*af.mean():.0f} %)")
        if only_keep + only_arm > 0:
            res = stats.binomtest(only_keep, only_keep + only_arm, 0.5)
            print(f"  discordant: keep-only {only_keep}, {arm}-only {only_arm}; "
                  f"McNemar exact p = {res.pvalue:.2e}")

        for metric, unit in (("J", ""), ("dist", " m")):
            a = df[f"{arm}_{metric}"].values
            k = df[f"keep_{metric}"].values
            diff = a - k
            w = stats.wilcoxon(a, k)
            print(f"post-switch {metric}: keep {np.mean(k):6.2f} "
                  f"(median {np.median(k):6.2f}), {arm} {np.mean(a):6.2f} "
                  f"(median {np.median(a):6.2f}){unit}")
            print(f"  paired diff median {np.median(diff):+.2f}{unit}, "
                  f"{arm} better in {np.sum(diff > 0)}/{n} seeds, "
                  f"Wilcoxon p = {w.pvalue:.2e}")

        ok = ~kf & ~df[f"{arm}_fell"].astype(bool)
        if ok.sum() > 10:
            a = df.loc[ok, f"{arm}_J"].values
            k = df.loc[ok, "keep_J"].values
            w = stats.wilcoxon(a, k)
            print(f"post-switch J, no-fall pairs only (n={ok.sum()}): "
                  f"keep {k.mean():.2f}, {arm} {a.mean():.2f}, "
                  f"Wilcoxon p = {w.pvalue:.2e}")

    # ── Figure: primary comparison (keep vs switch) ──────────────────────
    C_KEEP, C_SWITCH, C_FAST = "#2a78d6", "#eb6834", "#8a8984"
    arm = "switch"
    af = df[f"{arm}_fell"].astype(bool)
    jk, ja = df["keep_J"].values, df[f"{arm}_J"].values
    dk, da = df["keep_dist"].values, df[f"{arm}_dist"].values
    wJ = stats.wilcoxon(ja, jk)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    ax = axes[0]
    lo = min(jk.min(), ja.min()) - 2
    hi = max(jk.max(), ja.max()) + 2
    ax.plot([lo, hi], [lo, hi], color="#8a8984", lw=1, ls="--", zorder=1)
    colors = np.where(af, C_SWITCH, np.where(kf, C_KEEP, "#52514e"))
    ax.scatter(jk, ja, s=22, c=colors, alpha=0.75, lw=0, zorder=2)
    ax.set_xlabel("keep flat-optimal: objective J on slope")
    ax.set_ylabel("switch to sloped-optimal: objective J on slope")
    ax.set_title(f"Post-switch objective J, {n} paired seeds\n"
                 f"(falls scored J = −50; Wilcoxon p = {wJ.pvalue:.1e})",
                 fontsize=10)
    ax.text(0.03, 0.95, f"above line: switch better\n"
            f"({np.sum(ja > jk)}/{n} seeds)",
            transform=ax.transAxes, va="top", fontsize=9, color="#52514e")
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["keep\nflat-optimal", "switch to\nsloped-optimal"]
    bars = [100 * kf.mean(), 100 * af.mean()]
    cols = [C_KEEP, C_SWITCH]
    if "switch_fast" in arms:
        labels.append("switch to\nfastest climber")
        bars.append(100 * df["switch_fast_fell"].astype(bool).mean())
        cols.append(C_FAST)
    ax.bar(labels, bars, color=cols, width=0.55)
    for i, b in enumerate(bars):
        ax.text(i, b + 0.5, f"{b:.0f} %", ha="center", fontsize=11)
    ax.set_ylabel("fall rate on slope [%]")
    ax.set_ylim(0, max(bars) * 1.25 + 5)
    ax.set_title("Falls within 10 s after terrain switch")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ddiff = da - dk
    wD = stats.wilcoxon(da, dk)
    bins = np.linspace(ddiff.min() - 0.1, ddiff.max() + 0.1, 25)
    ax.hist(ddiff, bins=bins, color="#52514e", alpha=0.85)
    ax.axvline(0, color="#8a8984", lw=1, ls="--")
    ax.axvline(np.median(ddiff), color=C_SWITCH, lw=2,
               label=f"median {np.median(ddiff):+.2f} m")
    ax.set_xlabel("distance on slope, switch − keep [m]")
    ax.set_ylabel("seeds")
    ax.set_title(f"Paired distance difference (Wilcoxon p = {wD.pvalue:.1e})",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Flat→10° slope at t = 10 s: hold flat-optimal CPG parameters "
                 "vs switch to sloped-optimal", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULTS_DIR, "switch_comparison.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["preselect", "run", "aggregate"])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.stage == "preselect":
        preselect(args.workers)
    elif args.stage == "run":
        run_main(args.seeds, args.workers)
    else:
        aggregate()


if __name__ == "__main__":
    main()
