"""Shared analysis + figures for the continual event-adaptation experiments.

Loads the `results/continual_events.csv` written by either experiment's
`run_experiment.py` and produces the paper figures:

  method_comparison.<ext>  fall rate / stability (RMS body tilt) / distance per
                           method, mean +/- SEM across seeds (the headline);
  learning_curve.<ext>     fall rate vs event ordinal within the trial -- shows
                           the search-based arms thinning out falls as memory
                           fills, while the anchors stay flat;
  timeseries_seed0.<ext>   forward-speed trace (seed 0) per method with the event
                           windows shaded and falls marked.

Both experiments share the same output schema, so this module is experiment-
agnostic; each folder's `analyze.py` just passes its results dir + a couple of
labels (the event word, the plot title).

Palette: an Okabe-Ito-derived categorical set, fixed order, validated
colourblind-safe on the four adapting arms (worst adjacent CVD dE 37 >> 12
floor). `noadapt` is a deliberate neutral gray (lower anchor); identity is
reinforced by distinct markers + line styles and direct value labels so it never
rests on colour alone.
"""

import csv
import os

import numpy as np

METHOD_ORDER = ["noadapt", "grid", "bo", "safegp", "oracle"]
LABELS = {"noadapt": "No adaptation", "grid": "Grid search",
          "bo": "Bayesian opt.", "safegp": "Safe GP (ours)", "oracle": "Oracle"}
PALETTE = {"noadapt": "#999999", "grid": "#E69F00", "bo": "#56B4E9",
           "safegp": "#D55E00", "oracle": "#009E73"}
MARKERS = {"noadapt": "o", "grid": "s", "bo": "^", "safegp": "D", "oracle": "*"}
LINESTYLES = {"noadapt": ":", "grid": "-", "bo": "-", "safegp": "-", "oracle": "--"}


# ── load ─────────────────────────────────────────────────────────────────────

def load_events(results_dir):
    """Read continual_events.csv into a list of dicts with typed fields."""
    path = os.path.join(results_dir, "continual_events.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run run_experiment.py first")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            def fnum(k):
                v = r.get(k, "")
                return float(v) if v not in ("", None) else np.nan
            rows.append(dict(method=r["method"], seed=int(r["seed"]),
                             event=int(r["event"]), fell=int(r["fell"]),
                             V=fnum("V"), tilt_rms=fnum("tilt_rms"),
                             dist=fnum("dist"), trial_dist=fnum("trial_dist"),
                             false_alarm=int(r["false_alarm"])))
    return rows


def methods_present(rows):
    present = {r["method"] for r in rows}
    return [m for m in METHOD_ORDER if m in present]


# ── per-method / per-seed aggregation ────────────────────────────────────────

def _per_seed(rows, method):
    """Per-seed metrics for one method (real events only): fall rate, mean
    surviving tilt, trial distance. Returns dict seed -> (fr, tilt, dist)."""
    seeds = sorted({r["seed"] for r in rows if r["method"] == method})
    out = {}
    for s in seeds:
        real = [r for r in rows if r["method"] == method and r["seed"] == s
                and not r["false_alarm"]]
        if not real:
            continue
        fr = float(np.mean([r["fell"] for r in real]))
        surv = [r["tilt_rms"] for r in real if not r["fell"]
                and np.isfinite(r["tilt_rms"])]
        tilt = float(np.mean(surv)) if surv else np.nan
        dist = float([r["trial_dist"] for r in real][0])
        out[s] = (fr, tilt, dist)
    return out


def _mean_sem(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return np.nan, np.nan
    m = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, sem


def summary_table(rows):
    """Per-method (mean, sem) for fall rate, surviving tilt, trial distance."""
    tab = {}
    for m in methods_present(rows):
        ps = _per_seed(rows, m)
        frs = [v[0] for v in ps.values()]
        tilts = [v[1] for v in ps.values()]
        dists = [v[2] for v in ps.values()]
        tab[m] = dict(fall_rate=_mean_sem(frs), tilt=_mean_sem(tilts),
                      dist=_mean_sem(dists), n_seeds=len(ps),
                      n_events=len([r for r in rows if r["method"] == m
                                    and not r["false_alarm"]]))
    return tab


def summary_dataframe(rows):
    """Per-method summary as a pandas DataFrame (nice inline display in a
    notebook): fall rate [%], surviving RMS tilt [deg], distance [m], each as
    mean and SEM across seeds."""
    import pandas as pd
    tab = summary_table(rows)
    recs = []
    for m in methods_present(rows):
        t = tab[m]
        recs.append({
            "method": LABELS[m], "seeds": t["n_seeds"], "events": t["n_events"],
            "fall_rate_% ": 100.0 * t["fall_rate"][0],
            "fall_rate_sem": 100.0 * t["fall_rate"][1],
            "tilt_deg": t["tilt"][0], "tilt_sem": t["tilt"][1],
            "dist_m": t["dist"][0], "dist_sem": t["dist"][1],
        })
    return pd.DataFrame(recs).set_index("method").round(2)


# ── figures ──────────────────────────────────────────────────────────────────

def _style():
    # NB: no matplotlib.use() here -- respect whatever backend the caller set
    # (inline in a notebook, Agg if a headless caller selected it first).
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
    })
    return plt


def fig_comparison(rows, title, event_word, out_path=None):
    """3-panel headline: fall rate / surviving RMS tilt / trial distance per
    method, mean +/- SEM across seeds, with direct value labels. Saves to
    out_path if given; returns the matplotlib Figure (for inline display)."""
    plt = _style()
    methods = methods_present(rows)
    tab = summary_table(rows)
    x = np.arange(len(methods))
    colors = [PALETTE[m] for m in methods]
    hatch = ["///" if m == "noadapt" else None for m in methods]  # relief for gray

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    panels = [
        ("fall_rate", "fall rate", lambda v: v * 100.0, "%", "{:.0f}"),
        ("tilt", "stability: RMS body tilt", lambda v: v, "deg", "{:.1f}"),
        ("dist", "distance travelled", lambda v: v, "m", "{:.0f}"),
    ]
    for ax, (key, ylab, conv, unit, fmt) in zip(axes, panels):
        means = np.array([conv(tab[m][key][0]) for m in methods])
        sems = np.array([conv(tab[m][key][1]) for m in methods])
        bars = ax.bar(x, means, yerr=sems, capsize=3, color=colors,
                      edgecolor="#333333", linewidth=0.6, width=0.72,
                      error_kw=dict(lw=1.0, ecolor="#555555"))
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch(h)
        top = np.nanmax(means + np.nan_to_num(sems)) if len(means) else 1.0
        pad = 0.03 * (top if top else 1.0)
        for xi, mv, sv in zip(x, means, sems):
            if np.isfinite(mv):
                ax.text(xi, mv + (sv if np.isfinite(sv) else 0.0) + pad,
                        fmt.format(mv), ha="center", va="bottom", fontsize=8.5,
                        color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=30, ha="right",
                           fontsize=8.5)
        ax.set_ylabel(f"{ylab} [{unit}]")
        ax.set_title(ylab)
        ax.margins(y=0.18)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def fig_learning(rows, title, event_word, out_path=None):
    """Fall rate vs event ordinal within the trial (pooled across seeds). Search
    arms should thin out falls as memory fills; anchors stay flat. Saves to
    out_path if given; returns the Figure."""
    plt = _style()
    methods = methods_present(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for m in methods:
        real = [r for r in rows if r["method"] == m and not r["false_alarm"]]
        ords = sorted({r["event"] for r in real})
        xs, ys = [], []
        for e in ords:
            fe = [r["fell"] for r in real if r["event"] == e]
            if len(fe) >= 1:
                xs.append(e); ys.append(100.0 * float(np.mean(fe)))
        if not xs:
            continue
        # light smoothing over a width-3 window for readability
        ys = np.asarray(ys, float)
        if len(ys) >= 3:
            k = np.ones(3) / 3.0
            ys = np.convolve(ys, k, mode="same")
        ax.plot(xs, ys, marker=MARKERS[m], ls=LINESTYLES[m], color=PALETTE[m],
                lw=1.8, ms=5, mec="#333333", mew=0.4, label=LABELS[m])
    ax.set_xlabel(f"{event_word} event # within the trial")
    ax.set_ylabel("fall rate [%]")
    ax.set_ylim(-5, 105)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def fig_timeseries(results_dir, title, event_word, seed=0, out_path=None):
    """Seed-`seed` forward-speed trace per method, event windows shaded, falls
    marked. Returns None (no Figure) if the per-seed npz logs are absent."""
    plt = _style()
    log_dir = os.path.join(results_dir, "logs")
    methods = [m for m in METHOD_ORDER
               if os.path.exists(os.path.join(log_dir, f"{m}_seed{seed}.npz"))]
    if not methods:
        return None
    fig, axes = plt.subplots(len(methods), 1, figsize=(9, 1.7 * len(methods)),
                             sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        d = np.load(os.path.join(log_dir, f"{m}_seed{seed}.npz"))
        t, vx, shift, state = d["t"], d["vx"], d["shift"], d["state"]
        ax.axhspan(-10, -10, color="none")  # keep autoscale sane
        ax.fill_between(t, -0.6, 1.4, where=shift > 0.5, color="#D55E00",
                        alpha=0.10, lw=0)
        ax.fill_between(t, -0.6, 1.4, where=state > 0.5, color="#009E73",
                        alpha=0.10, lw=0)
        ax.plot(t, vx, lw=0.8, color=PALETTE[m])
        ax.axhline(0.5, color="#888888", ls=":", lw=0.7)
        ax.set_ylim(-0.6, 1.4)
        ax.set_ylabel("vx [m/s]", fontsize=8.5)
        ax.text(0.01, 0.86, LABELS[m], transform=ax.transAxes, fontsize=9,
                color=PALETTE[m], fontweight="bold", va="top")
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title(f"{title}\n(orange = {event_word} engaged, green = responding)",
                      fontsize=10)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


# ── orchestration ────────────────────────────────────────────────────────────

def _fmt(ms):
    m, s = ms
    return f"{m:.3g} +/- {s:.2g}" if np.isfinite(m) else "n/a"


def analyze(results_dir, title, event_word, ext="png"):
    """Load results, print a summary table, and write all three figures into
    results_dir/figures/. Returns the summary table dict."""
    rows = load_events(results_dir)
    tab = summary_table(rows)
    print(f"\n=== {title} ===")
    print(f"{'method':<14}{'seeds':>6}{'events':>8}{'fall rate':>16}"
          f"{'surv. tilt [deg]':>20}{'distance [m]':>18}")
    for m in methods_present(rows):
        t = tab[m]
        fr = (100.0 * t["fall_rate"][0], 100.0 * t["fall_rate"][1])
        print(f"{LABELS[m]:<14}{t['n_seeds']:>6}{t['n_events']:>8}"
              f"{_fmt(fr) + ' %':>16}{_fmt(t['tilt']):>20}{_fmt(t['dist']):>18}")

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    outs = []
    p1 = os.path.join(fig_dir, f"method_comparison.{ext}")
    fig_comparison(rows, title, event_word, out_path=p1); outs.append(p1)
    p2 = os.path.join(fig_dir, f"learning_curve.{ext}")
    fig_learning(rows, title, event_word, out_path=p2); outs.append(p2)
    p3 = os.path.join(fig_dir, f"timeseries_seed0.{ext}")
    if fig_timeseries(results_dir, title, event_word, out_path=p3) is not None:
        outs.append(p3)
    plt.close("all")
    print("\nsaved figures:")
    for o in outs:
        print(f"  {o}")
    return tab
