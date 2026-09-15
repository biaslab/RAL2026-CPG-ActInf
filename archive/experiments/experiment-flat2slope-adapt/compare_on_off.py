"""Aggregate the attitude-feedback ON-vs-OFF comparison into a fall-rate table.

Reads the per-condition manifests written by run_experiment.py:
    manifest.csv           CE trigger, feedback ON
    manifest_noafb.csv     CE trigger, feedback OFF
    manifest_dt.csv        DT trigger, feedback ON
    manifest_dt_noafb.csv  DT trigger, feedback OFF
and prints, per (trigger, slope, arm), the fall rate [%], mean upright time [s],
and mean windowed criterion V-bar, side by side for feedback ON vs OFF.

Usage (from repo root):
    python experiment-flat2slope-adapt/compare_on_off.py
"""

import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")

CONDS = [("ce", "manifest.csv", "manifest_noafb.csv"),
         ("dt", "manifest_dt.csv", "manifest_dt_noafb.csv")]
ARMS = ["noadapt", "oracle", "grid", "bo", "marxefe"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _agg(rows, slope, arm):
    """(-> n, falls%, mean t_surv, mean V-bar) for one slope/arm subset."""
    sub = [r for r in rows if r["method"] == arm
           and abs(float(r["slope_deg"]) - slope) < 1e-6]
    if not sub:
        return None
    n = len(sub)
    falls = sum(int(r["fell"]) for r in sub)
    ts = [float(r["t_surv"]) for r in sub if r.get("t_surv") not in ("", None)]
    vs = [float(r["mean_J"]) for r in sub if r.get("mean_J") not in ("", None)]
    mt = sum(ts) / len(ts) if ts else float("nan")
    mv = sum(vs) / len(vs) if vs else float("nan")
    return n, 100.0 * falls / n, mt, mv


def main():
    for trig, on_f, off_f in CONDS:
        on = _load(os.path.join(RESULTS, on_f))
        off = _load(os.path.join(RESULTS, off_f))
        if on is None and off is None:
            continue
        slopes = sorted({float(r["slope_deg"]) for r in (on or off)})
        print(f"\n===== trigger = {trig.upper()}  (feedback ON vs OFF) =====")
        for slope in slopes:
            print(f"\n  slope {slope:g} deg"
                  f"    {'falls% ON/OFF':>16s} {'uptime ON/OFF':>16s} {'Vbar ON/OFF':>16s}")
            for arm in ARMS:
                a = _agg(on, slope, arm) if on else None
                b = _agg(off, slope, arm) if off else None
                if a is None and b is None:
                    continue
                fa = f"{a[1]:.0f}" if a else "-"
                fb = f"{b[1]:.0f}" if b else "-"
                ta = f"{a[2]:.1f}" if a else "-"
                tb = f"{b[2]:.1f}" if b else "-"
                va = f"{a[3]:+.2f}" if a else "-"
                vb = f"{b[3]:+.2f}" if b else "-"
                nstr = f"(n={a[0] if a else (b[0] if b else 0)})"
                print(f"    {arm:9s} {nstr:7s} {fa:>7s}/{fb:<7s} "
                      f"{ta:>7s}/{tb:<7s} {va:>7s}/{vb:<7s}")


if __name__ == "__main__":
    main()
