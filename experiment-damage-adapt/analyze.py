"""Aggregate + plot the leg-damage online-adaptation runs.

Reads `results/manifest.csv` and the per-run `results/runs/dm_*.npz` written by
`run_experiment.py` and produces, in `results/figures/`:

  * summary.png       -- survival distance / fall rate / phase-2 velocity /
                         phase-2 tip-over / phase-2 mechanical power, bars over
                         seeds, one bar per method;
  * trace_seed{S}.png -- one seed: vx and tip deviation vs time for every
                         method, with the damage onset marked; shows no-adapt
                         degrading/falling while the adapters (and oracle) hold.

A weakened leg is a persistent mismatch, so besides falls the phase-2 velocity /
power gap is the discriminating signal (no-adapt "limps along inefficiently").

Usage (from repo root):
    python experiment-damage-adapt/analyze.py            # ce runs
    python experiment-damage-adapt/analyze.py --trigger cusum
    python experiment-damage-adapt/analyze.py --seed 3   # trace figure for seed 3
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_HERE, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

METHODS = [("noadapt", "tab:gray"), ("grid", "tab:green"),
           ("bo", "tab:blue"), ("marxefe", "tab:orange"),
           ("oracle", "tab:red")]
LABEL = {"noadapt": "No-adapt (hold flat)", "grid": "Grid (windowed)",
         "bo": "BO (windowed)", "marxefe": "MARX-EFE",
         "oracle": "Oracle (per-phase fit)"}

DT = 0.01


def _files(trigger):
    suffix = "" if trigger == "ce" else f"_{trigger}"
    pat = os.path.join(RUNS_DIR, f"dm_seed*_{{m}}{suffix}.npz")
    return {m: sorted(glob.glob(pat.format(m=m)),
                      key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
            for m, _ in METHODS}


def _tip_dev(roll, pitch):
    return np.rad2deg(np.sqrt(np.asarray(roll) ** 2 + np.asarray(pitch) ** 2))


def aggregate(trigger):
    files = _files(trigger)
    summ = {m: {"dist": [], "fell": [], "vx2": [], "tip2": [], "pw2": [],
                "n_trig": []} for m, _ in METHODS}
    for m, _ in METHODS:
        for f in files[m]:
            d = np.load(f, allow_pickle=True)
            n = len(d["y"])
            t = np.asarray(d["t"])
            t_dmg = float(d["t_shift"])
            ph2 = (t >= t_dmg) if t_dmg >= 0 else np.zeros(n, bool)
            tip = _tip_dev(d["roll"], d["pitch"])
            summ[m]["dist"].append(float(d["y"][-1]) if n else 0.0)
            summ[m]["fell"].append(bool(int(d["fell"])))
            summ[m]["vx2"].append(float(np.mean(np.asarray(d["vx"])[ph2]))
                                  if ph2.any() else np.nan)
            summ[m]["tip2"].append(float(np.mean(tip[ph2])) if ph2.any() else np.nan)
            summ[m]["pw2"].append(float(np.mean(np.asarray(d["power"])[ph2]))
                                  if ph2.any() else np.nan)
            summ[m]["n_trig"].append(int(len(d["fire_steps"])))
    return summ, files


def print_table(summ, files):
    n = max((len(files[m]) for m, _ in METHODS), default=0)
    print(f"\n=== leg-damage: survival / stability over {n} seeds ===")
    print(f"  {'method':24s} {'dist[m]':>12s} {'falls':>7s} {'vx2[m/s]':>10s} "
          f"{'tip2[deg]':>10s} {'P2[W]':>8s} {'trig':>6s}")
    for m, _ in METHODS:
        s = summ[m]
        if not s["dist"]:
            continue
        nf = len(files[m])
        print(f"  {LABEL[m]:24s} "
              f"{np.mean(s['dist']):6.1f}±{np.std(s['dist']):4.1f} "
              f"{sum(s['fell']):3d}/{nf:<3d} "
              f"{np.nanmean(s['vx2']):9.2f} "
              f"{np.nanmean(s['tip2']):9.1f} "
              f"{np.nanmean(s['pw2']):7.0f} "
              f"{np.mean(s['n_trig']):5.1f}")


def plot_summary(summ, files, trigger):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    names = [m for m, _ in METHODS if summ[m]["dist"]]
    labs = [LABEL[m] for m in names]
    cols = [c for m, c in METHODS if summ[m]["dist"]]
    nseed = max(len(files[m]) for m in names)

    fig, ax = plt.subplots(1, 5, figsize=(21, 4.2))
    ax[0].bar(labs, [np.mean(summ[m]["dist"]) for m in names],
              yerr=[np.std(summ[m]["dist"]) for m in names], color=cols,
              alpha=0.85, capsize=4)
    ax[0].set_ylabel("distance travelled [m]")
    ax[0].set_title("Survival distance (higher=better)")
    ax[1].bar(labs, [100.0 * sum(summ[m]["fell"]) / len(files[m]) for m in names],
              color=cols, alpha=0.85)
    ax[1].set_ylabel("fall rate [%]"); ax[1].set_title("Falls (lower=better)")
    ax[2].bar(labs, [np.nanmean(summ[m]["vx2"]) for m in names],
              yerr=[np.nanstd(summ[m]["vx2"]) for m in names], color=cols,
              alpha=0.85, capsize=4)
    ax[2].axhline(0.5, color="k", ls=":", lw=1)
    ax[2].set_ylabel("phase-2 mean vx [m/s]")
    ax[2].set_title("Post-damage speed (target 0.5)")
    ax[3].bar(labs, [np.nanmean(summ[m]["tip2"]) for m in names],
              yerr=[np.nanstd(summ[m]["tip2"]) for m in names], color=cols,
              alpha=0.85, capsize=4)
    ax[3].set_ylabel("phase-2 mean tip dev [deg]")
    ax[3].set_title("Post-damage tip-over (lower=better)")
    ax[4].bar(labs, [np.nanmean(summ[m]["pw2"]) for m in names],
              yerr=[np.nanstd(summ[m]["pw2"]) for m in names], color=cols,
              alpha=0.85, capsize=4)
    ax[4].set_ylabel("phase-2 mech. power [W]")
    ax[4].set_title("Post-damage power (lower=better)")
    for a in ax:
        a.grid(True, axis="y", alpha=0.3)
        a.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle(f"Online CPG adaptation under leg damage ({nseed} seeds, "
                 f"trigger={trigger})", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"summary{'' if trigger=='ce' else '_'+trigger}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def _load_seed(files, seed, trigger):
    out = {}
    for m, _ in METHODS:
        for f in files[m]:
            if int(re.search(r"seed(\d+)", f).group(1)) == seed:
                out[m] = np.load(f, allow_pickle=True)
    return out


def plot_traces(files, seed, trigger):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = _load_seed(files, seed, trigger)
    if not runs:
        print(f"(no runs for seed {seed}); skipping trace figure")
        return
    ref = next(iter(runs.values()))
    t_dmg = float(ref["t_shift"])

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for m, c in METHODS:
        if m in runs:
            r = runs[m]
            ax[0].plot(r["t"], r["vx"], color=c, lw=1.1, label=LABEL[m])
            ax[1].plot(r["t"], _tip_dev(r["roll"], r["pitch"]), color=c, lw=1.1)
    for a in ax:
        if t_dmg >= 0:
            a.axvline(t_dmg, color="k", ls="--", lw=1.2, label="damage onset")
        a.grid(True, alpha=0.3)
    ax[0].axhline(0.5, color="k", ls=":", lw=1)
    ax[0].set_ylabel("forward vx [m/s]"); ax[0].set_ylim(-0.4, 1.2)
    ax[0].set_title(f"Leg damage seed {seed}: velocity (top) & tip deviation "
                    f"(bottom), damage at t={t_dmg:g}s")
    ax[0].legend(loc="upper right", fontsize=8, ncol=2)
    ax[1].set_ylabel("tip dev √(roll²+pitch²) [deg]")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylim(0, 90)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"trace_seed{seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="ce")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed to draw the per-run trace figure for")
    args = ap.parse_args()
    summ, files = aggregate(args.trigger)
    if not any(files[m] for m, _ in METHODS):
        raise SystemExit(f"no {args.trigger} runs found in {RUNS_DIR}")
    print_table(summ, files)
    plot_summary(summ, files, args.trigger)
    plot_traces(files, args.seed, args.trigger)


if __name__ == "__main__":
    main()
