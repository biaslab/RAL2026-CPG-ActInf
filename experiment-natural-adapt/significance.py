"""Significance report for the natural-transect batch: per-arm falls and survival
distance, with paired tests of MARX-EFE against the references.

Reads a manifest (default the CUSUM one) and prints, over all seeds:
  * fall rate, reached-end rate, mean triggers/run, mean survival distance, mean J;
  * McNemar exact test on the paired fall outcomes (MARX-EFE vs each other arm);
  * Wilcoxon signed-rank on the paired survival distances (MARX-EFE vs each arm).

Distance is the discriminating metric on this terrain (almost everything eventually
falls), and the tests are PAIRED because every arm shares the same per-seed terrain.

Usage (from repo root):
    python experiment-natural-adapt/significance.py                 # cusum manifest
    python experiment-natural-adapt/significance.py --trigger dt
"""

import argparse
import csv
import os
from math import comb

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")
ARMS = ["noadapt", "oracle", "grid", "bo", "marxefe"]
# arms whose fall/distance we test against the no-adapt reference (paired)
VS_NOADAPT = ["oracle", "marxefe", "grid", "bo"]


def _load(trigger):
    suffix = "" if trigger == "ce" else f"_{trigger}"
    path = os.path.join(RESULTS, f"manifest{suffix}.csv")
    rows = list(csv.DictReader(open(path)))
    seeds = sorted({int(r["seed"]) for r in rows})
    by = {m: {} for m in ARMS}
    for r in rows:
        by[r["method"]][int(r["seed"])] = r
    return by, seeds, path


def _mcnemar_exact(a_only, b_only):
    """Two-sided exact McNemar p on discordant counts (binomial, p=0.5)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="cusum", choices=["ce", "dt", "cusum"])
    args = ap.parse_args()
    by, seeds, path = _load(args.trigger)
    n = len(seeds)
    print(f"manifest: {os.path.relpath(path)}  ({n} seeds)\n")

    arms = [m for m in ARMS if by.get(m)]         # only arms present in the manifest
    fell = {m: np.array([int(by[m][s]["fell"]) for s in seeds]) for m in arms}
    dist = {m: np.array([float(by[m][s]["dist"]) for s in seeds]) for m in arms}
    trig = {m: np.array([int(by[m][s]["n_triggers"]) for s in seeds]) for m in arms}
    endr = {m: np.array([int(by[m][s]["reached_end"]) for s in seeds]) for m in arms}

    print(f"  {'arm':9s} {'falls':>8s} {'end':>7s} {'trig/run':>9s} "
          f"{'dist[m]':>13s}")
    for m in arms:
        print(f"  {m:9s} {fell[m].sum():3d}/{n:<4d} {endr[m].sum():3d}/{n:<3d} "
              f"{trig[m].mean():9.1f} {dist[m].mean():6.1f}±{dist[m].std():<5.1f}")

    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    if "noadapt" not in arms:
        return
    na_f, na_d = fell["noadapt"], dist["noadapt"]
    print("\n  vs NO-ADAPT (paired over seeds):")
    for m in VS_NOADAPT:
        if m not in arms:
            continue
        saves = int(np.sum(na_f.astype(bool) & ~fell[m].astype(bool)))   # noadapt fell, m survived
        loses = int(np.sum(fell[m].astype(bool) & ~na_f.astype(bool)))   # m fell, noadapt survived
        p_mc = _mcnemar_exact(saves, loses)
        d = dist[m] - na_d
        if wilcoxon is not None and np.any(d != 0):
            _, p_w = wilcoxon(dist[m], na_d)
        else:
            p_w = float("nan")
        print(f"    {m:8s} falls {fell[m].sum():d} vs {na_f.sum():d} "
              f"(saves {saves}, loses {loses}, McNemar p={p_mc:.3f})  |  "
              f"dist {d.mean():+5.1f}m median {np.median(d):+5.1f}m Wilcoxon p={p_w:.3f}")


if __name__ == "__main__":
    main()
