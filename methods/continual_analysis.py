"""Shared analysis + figures for the continual event-adaptation experiments.

Loads the `results/continual_events.csv` written by either experiment's
`run_experiment.py` and produces the paper figures:

  method_comparison.<ext>  falls per bout / stability (RMS body tilt) / distance
                           per method, mean +/- SEM across seeds (the headline);
  falls_over_time.<ext>    mean cumulative falls vs time per method -- no-adapt
                           climbs steadily, a method that learns a recovery gait
                           plateaus, the oracle stays flat;
  timeseries_seed0.<ext>   forward-speed trace (seed 0) per method with the event
                           windows shaded.

NB: the event persists until a fall (see run_experiment), so the headline metric
is FALLS PER BOUT, not a per-event fall fraction. Distance here is DISTANCE UNDER
THE PERTURBATION (sum of per-event onset->fall/end progress), NOT total bout
distance -- otherwise a failing method is rewarded for the fast healthy walking
between its falls.

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

# fixed (non-parameter) columns; anything else in the header is a searched CPG
# parameter (the candidate gait's free dims), captured per event as `params`.
_FIXED_COLS = {"method", "seed", "event", "onset", "detect", "latency", "fell",
               "V", "tilt_rms", "dist", "false_alarm", "request_t",
               "compute_latency", "trial_dist"}


def load_events(results_dir):
    """Read continual_events.csv into a list of dicts with typed fields. The
    trailing parameter columns (the searched free dims) are captured as `params`
    (a float array) so the search dynamics can be analysed."""
    path = os.path.join(results_dir, "continual_events.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run run_experiment.py first")
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in (reader.fieldnames or []) if c not in _FIXED_COLS]
        for r in reader:
            def fnum(k):
                v = r.get(k, "")
                return float(v) if v not in ("", None) else np.nan
            rows.append(dict(method=r["method"], seed=int(r["seed"]),
                             event=int(r["event"]), fell=int(r["fell"]),
                             V=fnum("V"), tilt_rms=fnum("tilt_rms"),
                             dist=fnum("dist"), trial_dist=fnum("trial_dist"),
                             false_alarm=int(r["false_alarm"]),
                             # async-driver timing (may be absent in older CSVs):
                             onset=fnum("onset"), request_t=fnum("request_t"),
                             compute_latency=fnum("compute_latency"),
                             latency=fnum("latency"),
                             params=np.array([fnum(c) for c in param_cols], float)))
    return rows


def methods_present(rows):
    present = {r["method"] for r in rows}
    return [m for m in METHOD_ORDER if m in present]


# ── per-method / per-seed aggregation ────────────────────────────────────────

def _per_seed(rows, method):
    """Per-seed metrics for one method: FALLS PER BOUT (count), mean surviving
    RMS tilt, and DISTANCE UNDER THE PERTURBATION (sum of per-event forward
    progress from event onset to the fall/bout-end). The latter credits ground
    covered while coping with the fault -- unlike total bout distance, it does
    NOT reward a failing method for the fast healthy walking between its falls.
    Returns dict seed -> (n_falls, tilt, event_dist)."""
    seeds = sorted({r["seed"] for r in rows if r["method"] == method})
    out = {}
    for s in seeds:
        evs = [r for r in rows if r["method"] == method and r["seed"] == s]
        if not evs:
            continue
        n_falls = int(sum(r["fell"] for r in evs))
        surv = [r["tilt_rms"] for r in evs if not r["fell"]
                and np.isfinite(r["tilt_rms"])]
        tilt = float(np.mean(surv)) if surv else np.nan
        event_dist = float(sum(r["dist"] for r in evs if np.isfinite(r["dist"])))
        out[s] = (n_falls, tilt, event_dist)
    return out


def _mean_sem(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return np.nan, np.nan
    m = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, sem


def summary_table(rows):
    """Per-method (mean, sem) across seeds for falls-per-bout, surviving tilt,
    trial distance."""
    tab = {}
    for m in methods_present(rows):
        ps = _per_seed(rows, m)
        falls = [v[0] for v in ps.values()]
        tilts = [v[1] for v in ps.values()]
        dists = [v[2] for v in ps.values()]
        tab[m] = dict(falls=_mean_sem(falls), tilt=_mean_sem(tilts),
                      dist=_mean_sem(dists), n_seeds=len(ps),
                      n_falls_total=int(sum(falls)))
    return tab


def summary_dataframe(rows):
    """Per-method summary as a pandas DataFrame (nice inline display in a
    notebook): falls per bout, surviving RMS tilt [deg], distance [m], each as
    mean and SEM across seeds."""
    import pandas as pd
    tab = summary_table(rows)
    recs = []
    for m in methods_present(rows):
        t = tab[m]
        recs.append({
            "method": LABELS[m], "seeds": t["n_seeds"],
            "falls_per_bout": t["falls"][0], "falls_sem": t["falls"][1],
            "tilt_deg": t["tilt"][0], "tilt_sem": t["tilt"][1],
            "dist_m": t["dist"][0], "dist_sem": t["dist"][1],
        })
    return pd.DataFrame(recs).set_index("method").round(2)


# ── falls during search (the mechanism: destabilizing trials avoided) ─────────
#
# In this continual setting every fall is a candidate gait executed on the robot
# that then tipped it -- a "destabilizing trial". The thesis is that the safe
# agent avoids these by screening each candidate against a memory of past falls,
# instead of re-testing gaits it has already seen fall. We quantify that two ways:
#   (1) AVOIDABLE (repeat) falls -- the fraction of a bout's falls that re-crash a
#       region already known to fall (a candidate within `eps`, in the normalized
#       free-dim space, of an earlier failed candidate in the same bout). A memory
#       that works drives this toward zero; a memoryless optimizer re-probes.
#   (2) fall-rate DECAY -- falls in the first vs. second half of the bout. As the
#       memory fills, a learning method should fall less later; a non-learning one
#       stays flat.

def _param_normalizers(rows):
    P = np.array([r["params"] for r in rows
                  if r["params"].size and np.all(np.isfinite(r["params"]))], float)
    if P.size == 0:
        return None, None
    lo = P.min(axis=0)
    rng = np.where(P.max(axis=0) > lo, P.max(axis=0) - lo, 1.0)
    return lo, rng


def search_summary(rows, eps=0.15, bout_duration=None):
    """Per-method search dynamics (see block comment). `eps` is the normalized
    free-dim distance within which two candidates count as the same failed region;
    `bout_duration` (s) sets the early/late split (inferred from max onset if None).
    Returns dict method -> dict(falls, repeat_frac, falls_early, falls_late), each
    value a (mean, sem) across seeds."""
    lo, rng = _param_normalizers(rows)
    if bout_duration is None:
        ons = [r["onset"] for r in rows if np.isfinite(r["onset"])]
        bout_duration = max(ons) if ons else 0.0
    half = bout_duration / 2.0
    out = {}
    for m in methods_present(rows):
        seeds = sorted({r["seed"] for r in rows if r["method"] == m})
        falls, rep, early, late = [], [], [], []
        for s in seeds:
            evs = sorted([r for r in rows if r["method"] == m and r["seed"] == s],
                         key=lambda r: r["event"])
            failed, n_fall, n_rep, e_e, e_l = [], 0, 0, 0, 0
            for r in evs:
                if not r["fell"]:
                    continue
                n_fall += 1
                if lo is not None and np.all(np.isfinite(r["params"])):
                    c = (r["params"] - lo) / rng
                    if failed and min(np.linalg.norm(c - f) for f in failed) < eps:
                        n_rep += 1
                    failed.append(c)
                if np.isfinite(r["onset"]):
                    if r["onset"] < half:
                        e_e += 1
                    else:
                        e_l += 1
            falls.append(n_fall)
            rep.append(n_rep / n_fall if n_fall else np.nan)
            early.append(e_e)
            late.append(e_l)
        out[m] = dict(falls=_mean_sem(falls), repeat_frac=_mean_sem(rep),
                      falls_early=_mean_sem(early), falls_late=_mean_sem(late))
    return out


def search_dataframe(rows, eps=0.15, bout_duration=None):
    """Search-dynamics summary as a pandas DataFrame: falls per bout, avoidable
    (repeat) fall fraction, and first- vs second-half falls."""
    import pandas as pd
    ss = search_summary(rows, eps=eps, bout_duration=bout_duration)
    recs = []
    for m in methods_present(rows):
        t = ss[m]
        recs.append({
            "method": LABELS[m],
            "falls_per_bout": t["falls"][0],
            "avoidable_fall_frac": t["repeat_frac"][0],
            "falls_first_half": t["falls_early"][0],
            "falls_second_half": t["falls_late"][0],
        })
    return pd.DataFrame(recs).set_index("method").round(3)


# ── async timing (detection vs. optimizer compute latency) ───────────────────

def latency_summary(rows):
    """Per-method async-driver timing. The responder runs OUT OF PROCESS, so a
    detected event incurs (i) a detection/trigger latency (event onset -> CUSUM
    fire = `request_t - onset`) and (ii) an optimizer COMPUTE latency (`request_t`
    -> the gait actually applied = `compute_latency`), during which the robot
    keeps walking on the un-adapted gait. Returns dict method -> dict with
    det_latency (mean, sem) [s], compute_latency (median, p90, max) [s], and the
    event count. Rows lacking the columns (older CSVs) are skipped gracefully."""
    out = {}
    for m in methods_present(rows):
        evs = [r for r in rows if r["method"] == m and not r["false_alarm"]]
        det = np.array([r["request_t"] - r["onset"] for r in evs
                        if np.isfinite(r.get("request_t", np.nan))
                        and np.isfinite(r.get("onset", np.nan))], float)
        cl = np.array([r["compute_latency"] for r in evs
                       if np.isfinite(r.get("compute_latency", np.nan))], float)
        out[m] = dict(
            det_latency=(_mean_sem(det) if len(det) else (np.nan, np.nan)),
            cl_median=float(np.median(cl)) if len(cl) else np.nan,
            cl_p90=float(np.percentile(cl, 90)) if len(cl) else np.nan,
            cl_max=float(np.max(cl)) if len(cl) else np.nan,
            n_events=len(evs))
    return out


def latency_dataframe(rows):
    """Per-method async timing as a pandas DataFrame: detection latency [s]
    (event onset -> trigger) and optimizer compute latency [s] (trigger -> gait
    applied; median / p90 / max)."""
    import pandas as pd
    lat = latency_summary(rows)
    recs = []
    for m in methods_present(rows):
        t = lat[m]
        recs.append({
            "method": LABELS[m], "events": t["n_events"],
            "det_latency_s": t["det_latency"][0], "det_sem_s": t["det_latency"][1],
            "compute_lat_median_s": t["cl_median"], "compute_lat_p90_s": t["cl_p90"],
            "compute_lat_max_s": t["cl_max"],
        })
    return pd.DataFrame(recs).set_index("method").round(3)


def overall_detection(rows):
    """Mean detection latency [s] over ALL detected real events (any method) and
    the total false-alarm count -- the paper's single 'detection averaged X s with
    no false alarms' statement."""
    det = np.array([r["request_t"] - r["onset"] for r in rows
                    if not r["false_alarm"]
                    and np.isfinite(r.get("request_t", np.nan))
                    and np.isfinite(r.get("onset", np.nan))], float)
    n_false = int(sum(r["false_alarm"] for r in rows))
    return (float(np.mean(det)) if len(det) else np.nan, n_false, len(det))


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
        ("falls", "falls per bout", lambda v: v, "count", "{:.1f}"),
        ("tilt", "stability: RMS body tilt", lambda v: v, "deg", "{:.1f}"),
        ("dist", "distance under perturbation", lambda v: v, "m", "{:.0f}"),
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


def fig_falls_over_time(results_dir, title, event_word, out_path=None):
    """Cumulative falls vs time, mean across seeds per method (from the per-seed
    `logs/*.npz` `cum_falls` traces). No-adapt climbs steadily; a method that
    learns a recovery gait plateaus; the oracle stays flat. Returns the Figure,
    or None if the logs are absent."""
    plt = _style()
    log_dir = os.path.join(results_dir, "logs")
    methods = [m for m in METHOD_ORDER
               if _seed_logs(log_dir, m)]
    if not methods:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for m in methods:
        curves, ts = [], []
        for p in _seed_logs(log_dir, m):
            d = np.load(p)
            if "cum_falls" not in d.files:     # stale log from an older schema
                continue
            curves.append(d["cum_falls"]); ts.append(d["t"])
        if not curves:
            continue
        # align on the shortest trace (bouts are the same duration => same length)
        L = min(len(c) for c in curves)
        t = ts[0][:L]
        mean = np.mean([c[:L] for c in curves], axis=0)
        ax.plot(t, mean, ls=LINESTYLES[m], color=PALETTE[m], lw=2.0,
                label=LABELS[m])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cumulative falls (mean over seeds)")
    ax.set_title(f"{title}: falls accumulating over the bout")
    ax.legend(ncol=2, fontsize=8.5, loc="upper left")
    ax.margins(x=0.01)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def _seed_logs(log_dir, method):
    import glob
    return sorted(glob.glob(os.path.join(log_dir, f"{method}_seed*.npz")))


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
    print(f"{'method':<14}{'seeds':>6}{'falls/bout':>18}"
          f"{'surv. tilt [deg]':>20}{'distance [m]':>18}")
    for m in methods_present(rows):
        t = tab[m]
        print(f"{LABELS[m]:<14}{t['n_seeds']:>6}{_fmt(t['falls']):>18}"
              f"{_fmt(t['tilt']):>20}{_fmt(t['dist']):>18}")

    # async-driver timing: detection (trigger) vs optimizer compute latency
    lat = latency_summary(rows)
    det_mean, n_false, n_det = overall_detection(rows)
    print(f"\n  detection latency (onset->trigger): {det_mean:.2f} s over {n_det} "
          f"events, {n_false} false alarms")
    print(f"  {'method':<14}{'compute lat median':>20}{'p90':>8}{'max':>8}  [s]"
          f"  (trigger->gait applied, out-of-process)")
    for m in methods_present(rows):
        t = lat[m]
        print(f"  {LABELS[m]:<14}{t['cl_median']:>20.3f}{t['cl_p90']:>8.3f}"
              f"{t['cl_max']:>8.3f}")

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    outs = []
    p1 = os.path.join(fig_dir, f"method_comparison.{ext}")
    fig_comparison(rows, title, event_word, out_path=p1); outs.append(p1)
    p2 = os.path.join(fig_dir, f"falls_over_time.{ext}")
    if fig_falls_over_time(results_dir, title, event_word, out_path=p2) is not None:
        outs.append(p2)
    p3 = os.path.join(fig_dir, f"timeseries_seed0.{ext}")
    if fig_timeseries(results_dir, title, event_word, out_path=p3) is not None:
        outs.append(p3)
    plt.close("all")
    print("\nsaved figures:")
    for o in outs:
        print(f"  {o}")
    return tab
