"""Crossover with error bars: 3 param sets x 2 terrains x 32 reps."""
import json, sys
from multiprocessing import get_context
import numpy as np
from scipy import stats

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import run_experiment as rx

def job(arg):
    name, params, tname, rep = arg
    if tname == "flat":
        cfg = {"kind": "sloped", "slope_deg": rx.SLOPE_DEG,
               "slope_start_y": rx.FAR_SLOPE_Y, "n_cols": rx.N_COLS}
    else:
        cfg = {"kind": "sloped", "slope_deg": rx.SLOPE_DEG,
               "slope_start_y": 2.0, "n_cols": rx.N_COLS}
    log = rx.run_episode(cfg, seed=2000 + rep, params_start=np.asarray(params), duration=10.0)
    m = rx.window_metrics(log, 0, int(10.0 / rx.DT))
    m["fell"] = bool(log["fell"])
    if m["fell"]:
        m["J"] = -50.0
    return name, tname, rep, m

def main():
    with open(rx.SELECTED_JSON) as f:
        sel = json.load(f)
    sets = {"flat-opt": sel["flat"]["params"],
            "sloped-opt": sel["sloped"]["params"],
            "sloped-fast": sel["sloped_fast"]["params"]}
    R = 32
    jobs = [(n, p, t, r) for n, p in sets.items()
            for t in ("flat", "sloped") for r in range(R)]
    ctx = get_context("spawn")
    with ctx.Pool(10, maxtasksperchild=8) as pool:
        res = pool.map(job, jobs)

    M = {}
    print(f"\n{'param set':12s} {'terrain':8s} {'falls':>7s} {'dist [m]':>16s} {'mean vx':>14s} {'J':>16s}")
    for n in sets:
        for t in ("flat", "sloped"):
            ms = [m for (nn, tt, _, m) in res if nn == n and tt == t]
            M[(n, t)] = ms
            falls = sum(m["fell"] for m in ms)
            d = [m["dist"] for m in ms]; v = [m["mean_vx"] for m in ms]
            J = [m["J"] for m in ms]
            print(f"{n:12s} {t:8s} {falls:>5d}/{R} "
                  f"{np.mean(d):8.2f} ± {np.std(d):4.2f} "
                  f"{np.mean(v):7.2f} ± {np.std(v):4.2f} "
                  f"{np.mean(J):8.2f} ± {np.std(J):4.2f}")

    print("\nflat-opt vs sloped-opt, J per terrain (Mann-Whitney):")
    for t in ("flat", "sloped"):
        a = [m["J"] for m in M[("flat-opt", t)]]
        b = [m["J"] for m in M[("sloped-opt", t)]]
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {t:8s}: flat-opt {np.mean(a):5.2f} vs sloped-opt {np.mean(b):5.2f}, "
              f"p = {u.pvalue:.2e}")

if __name__ == "__main__":
    main()
