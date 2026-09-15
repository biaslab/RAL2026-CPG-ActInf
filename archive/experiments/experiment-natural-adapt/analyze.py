"""Aggregate + plot the natural-transect online-adaptation runs.

Reads `results/manifest.csv` and the per-run `results/runs/*.npz` written by
`run_experiment.py` and produces, in `results/figures/`:

  * summary.png          -- survival distance / fall rate / bands crossed /
                            velocity tracking / tip-over, bars over seeds;
  * velocity_trace.png   -- one seed: vx vs forward position, bands tinted by
                            surface type;
  * attitude_recovery.png-- one seed: tip deviation sqrt(roll^2+pitch^2) vs
                            position, with band transitions marked;
  * trigger_trace.png    -- one seed: the prediction-error ratio for MARX-EFE
                            with the fire markers (shows re-triggering per band).

On a natural transect almost everything eventually falls, so survival distance
(distance travelled before a fall / the terrain end) is the discriminating
metric and is shown first.

Usage (from repo root):
    python experiment-natural-adapt/analyze.py            # ce runs
    python experiment-natural-adapt/analyze.py --trigger dt
    python experiment-natural-adapt/analyze.py --seed 3   # trace figures for seed 3
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
           ("bo", "tab:blue"), ("marxefe", "tab:orange")]
LABEL = {"noadapt": "No-adapt (hold flat)", "grid": "Grid (windowed)",
         "bo": "BO (windowed)", "marxefe": "MARX-EFE"}

DT = 0.01


def _files(trigger):
    suffix = "" if trigger == "ce" else f"_{trigger}"
    pat = os.path.join(RUNS_DIR, f"nat_seed*_{{m}}{suffix}.npz")
    return {m: sorted(glob.glob(pat.format(m=m)),
                      key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
            for m, _ in METHODS}


def _tip_dev(roll, pitch):
    return np.rad2deg(np.sqrt(np.asarray(roll) ** 2 + np.asarray(pitch) ** 2))


def aggregate(trigger):
    files = _files(trigger)
    summ = {m: {"dist": [], "fell": [], "bands": [], "tip": [], "meanJ": [],
                "vx": [], "n_trig": []} for m, _ in METHODS}
    for m, _ in METHODS:
        for f in files[m]:
            d = np.load(f, allow_pickle=True)
            n = len(d["y"])
            t = d["t"]
            mtail = np.asarray(t) >= 3.4
            tip = _tip_dev(d["roll"], d["pitch"])
            ws = d["window_scores"]
            summ[m]["dist"].append(float(d["y"][-1]) if n else 0.0)
            summ[m]["fell"].append(bool(int(d["fell"])))
            by = d["bands_y"]
            summ[m]["bands"].append(int(np.sum((by > 0) & (by <= float(d["y"][-1])))))
            summ[m]["tip"].append(float(np.mean(tip[mtail])) if mtail.any() else np.nan)
            summ[m]["meanJ"].append(float(np.mean(ws)) if len(ws) else np.nan)
            summ[m]["vx"].append(float(np.mean(np.asarray(d["vx"])[mtail]))
                                 if mtail.any() else np.nan)
            summ[m]["n_trig"].append(int(len(d["fire_steps"])))
    return summ, files


def print_table(summ, files):
    n = max((len(files[m]) for m, _ in METHODS), default=0)
    print(f"\n=== natural transect: survival / stability over {n} seeds ===")
    print(f"  {'method':22s} {'dist[m]':>12s} {'falls':>7s} {'bands':>7s} "
          f"{'tip[deg]':>10s} {'meanJ':>8s} {'trig':>6s}")
    for m, _ in METHODS:
        s = summ[m]
        if not s["dist"]:
            continue
        nf = len(files[m])
        print(f"  {LABEL[m]:22s} "
              f"{np.mean(s['dist']):6.1f}±{np.std(s['dist']):4.1f} "
              f"{sum(s['fell']):3d}/{nf:<3d} "
              f"{np.mean(s['bands']):6.1f} "
              f"{np.nanmean(s['tip']):9.1f} "
              f"{np.nanmean(s['meanJ']):8.2f} "
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

    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
    ax[0].bar(labs, [np.mean(summ[m]["dist"]) for m in names],
              yerr=[np.std(summ[m]["dist"]) for m in names], color=cols,
              alpha=0.85, capsize=4)
    ax[0].set_ylabel("distance travelled [m]")
    ax[0].set_title("Survival distance (higher=better)")
    ax[1].bar(labs, [100.0 * sum(summ[m]["fell"]) / len(files[m]) for m in names],
              color=cols, alpha=0.85)
    ax[1].set_ylabel("fall rate [%]"); ax[1].set_title("Falls (lower=better)")
    ax[2].bar(labs, [np.mean(summ[m]["bands"]) for m in names],
              yerr=[np.std(summ[m]["bands"]) for m in names], color=cols,
              alpha=0.85, capsize=4)
    ax[2].set_ylabel("bands crossed"); ax[2].set_title("Bands crossed (higher=better)")
    ax[3].bar(labs, [np.nanmean(summ[m]["vx"]) for m in names],
              yerr=[np.nanstd(summ[m]["vx"]) for m in names], color=cols,
              alpha=0.85, capsize=4)
    ax[3].axhline(0.5, color="k", ls=":", lw=1)
    ax[3].set_ylabel("mean vx [m/s]"); ax[3].set_title("Forward speed (target 0.5)")
    ax[4].bar(labs, [np.nanmean(summ[m]["tip"]) for m in names],
              yerr=[np.nanstd(summ[m]["tip"]) for m in names], color=cols,
              alpha=0.85, capsize=4)
    ax[4].set_ylabel("mean tip dev [deg]"); ax[4].set_title("Tip-over (lower=better)")
    for a in ax:
        a.grid(True, axis="y", alpha=0.3)
        a.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle(f"Online CPG adaptation over natural terrain ({nseed} seeds, "
                 f"trigger={trigger})", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"summary{'' if trigger=='ce' else '_'+trigger}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def _load_seed(files, seed, trigger):
    """Return {method: npz} for one seed (only methods that have that seed)."""
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
    from matplotlib.patches import Patch
    from methods import terrain
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = _load_seed(files, seed, trigger)
    if not runs:
        print(f"(no runs for seed {seed}); skipping trace figures")
        return
    ref = next(iter(runs.values()))
    by = ref["bands_y"]; bn = ref["bands_name"]; reach = float(ref["reach"])
    ymax = max(float(r["y"][-1]) for r in runs.values())
    edges = list(by) + [reach]

    def shade(ax):
        for i, y0 in enumerate(by):
            ax.axvspan(y0, edges[i + 1],
                       color=terrain.NATURAL_COLORS[str(bn[i])], alpha=0.16, lw=0)

    band_legend = [Patch(facecolor=terrain.NATURAL_COLORS[nm], alpha=0.5,
                   label=f"{nm} (μ={terrain.NATURAL_SURFACES[nm][0]})")
                   for nm in ("grass", "gravel", "rocks", "river")]

    # velocity vs forward position
    fig, ax = plt.subplots(figsize=(12, 5)); shade(ax)
    for m, c in METHODS:
        if m in runs:
            ax.plot(runs[m]["y"], runs[m]["vx"], color=c, lw=1.1, label=LABEL[m])
    ax.axhline(0.5, color="k", ls=":", lw=1, label="target v*")
    ax.set_xlim(0, ymax + 0.5); ax.set_ylim(-0.3, 1.6)
    ax.set_xlabel("Forward position [m] (bands tinted by surface)")
    ax.set_ylabel("Forward velocity vx [m/s]")
    ax.set_title(f"Velocity tracking across natural terrain (seed {seed})")
    leg1 = ax.legend(loc="upper right", fontsize=8); ax.add_artist(leg1)
    ax.legend(handles=band_legend, loc="lower right", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    out = os.path.join(FIG_DIR, f"velocity_trace_seed{seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}")

    # tip-over deviation vs position
    fig, ax = plt.subplots(figsize=(12, 5)); shade(ax)
    for y0 in by:
        if 0 < y0 < ymax:
            ax.axvline(y0, color="gray", ls="--", lw=0.5, alpha=0.35)
    for m, c in METHODS:
        if m in runs:
            ax.plot(runs[m]["y"], _tip_dev(runs[m]["roll"], runs[m]["pitch"]),
                    color=c, lw=1.1, label=LABEL[m])
    ax.set_xlim(0, ymax + 0.5); ax.set_ylim(0, 90)
    ax.set_xlabel("Forward position [m] (dashed = band transitions)")
    ax.set_ylabel("Tip deviation √(roll²+pitch²) [deg]")
    ax.set_title(f"Tip-over stability / recovery at band transitions (seed {seed})")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    out = os.path.join(FIG_DIR, f"attitude_recovery_seed{seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}")

    # prediction-error ratio + fire markers (MARX-EFE run), showing re-triggering
    if "marxefe" in runs:
        r = runs["marxefe"]
        fig, ax = plt.subplots(figsize=(12, 4)); shade2 = ax
        for i, y0 in enumerate(by):
            ax.axvspan(y0, edges[i + 1],
                       color=terrain.NATURAL_COLORS[str(bn[i])], alpha=0.16, lw=0)
        ax.plot(r["y"], r["ratio"], color="tab:orange", lw=1.2,
                label="prediction-error ratio")
        ax.axhline(float(r["k_sigma"]), color="k", ls="--", lw=1,
                   label=f"threshold K={float(r['k_sigma']):g}")
        fs = np.asarray(r["fire_steps"], int)
        fs = fs[fs < len(r["y"])]
        ax.plot(r["y"][fs], r["ratio"][fs], "v", color="red", ms=8,
                label="event fired")
        ax.set_xlim(0, float(r["y"][-1]) + 0.5)
        ax.set_xlabel("Forward position [m]")
        ax.set_ylabel("error ratio (grass baseline = 1)")
        ax.set_title(f"MARX-EFE trigger re-firing per band transition (seed {seed})")
        ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
        out = os.path.join(FIG_DIR, f"trigger_trace_seed{seed}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", choices=["ce", "dt"], default="ce")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed to draw the per-run trace figures for")
    args = ap.parse_args()
    summ, files = aggregate(args.trigger)
    if not any(files[m] for m, _ in METHODS):
        raise SystemExit(f"no {args.trigger} runs found in {RUNS_DIR}")
    print_table(summ, files)
    plot_summary(summ, files, args.trigger)
    plot_traces(files, args.seed, args.trigger)


if __name__ == "__main__":
    main()
