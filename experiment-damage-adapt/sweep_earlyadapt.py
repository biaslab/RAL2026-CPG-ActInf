"""Culminating test of the pure-online-adaptation paradigm: can MARX-EFE DISCOVER
a recovery gait if it (a) reacts at the ONSET of prediction error instead of
waiting for the accumulated cusum trigger, and (b) keeps exploring persistently
through the long SAFE window of a gradual damage ramp?

Rationale: the persistent-excitation sweep showed the agent migrates toward the
stable region (d_opt 0.73->0.62) but stalls, because even under a 20 s gradual
ramp the cusum fires late (~ramp end) -- the whole mildly-damaged, safe early
descent is wasted sitting at the incumbent. Here we force the reaction at the
ramp start (force_trigger_t = t_damage), so the agent has the entire descent to
re-identify the changed dynamics and move, and combine it with persistent
excitation and a wider control-prior trust region (so it can cover the large
coupling/b move the recovery needs). No gait is ever supplied.

Grid: MARX-EFE under a 20 s gradual ramp, forced reaction at ramp start, over
control-prior scale x persist std; noadapt / oracle references. d_opt_min tracks
how close the agent's own parameters get to the (never-supplied) stable gait.

Usage (from repo root):
    python experiment-damage-adapt/sweep_earlyadapt.py --seeds 6 --workers 12
"""

import argparse
import csv
import importlib.util
import itertools
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
SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_earlyadapt.csv")
RAMP_END = 35.0


def _dm():
    spec = importlib.util.spec_from_file_location(
        "dm_ea", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _job(args):
    (seed, arm, scale, persist_std, ramp_t, force_start, trigger, duration) = args
    dm = _dm()
    f2s = dm.f2s
    f2s._limit_threads()
    f2s.MARX_CONTROL_PRIOR_SCALE = float(scale)
    f2s.MARX_GOAL_VEL_STD = float(dm.MARX_VEL_STD)
    f2s.MarxEFE.FORGETTING = float(dm.MARX_FORGETTING)
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = os.environ.get("CPG_ATTITUDE_FB", "1") != "0"
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    lo, hi = bl.numpy(), bu.numpy()
    box = (lo, hi); rng = hi - lo

    k_eff = 1.0 if trigger in ("dt", "cusum") else float(f2s.K_DEFAULT)
    incumbent = dm.load_incumbent()
    dc = dm.damage_defaults(duration)
    dc["ramp_t"] = float(ramp_t)
    dc["t_damage"] = float(RAMP_END) - float(ramp_t)
    if arm == "marxefe":
        train = dict(t0=float(dm.TRAIN_T0), std=float(dm.TRAIN_STD),
                     tau=float(dm.TRAIN_TAU))
        if persist_std > 0:
            train["persist"] = True
            train["persist_std"] = float(persist_std)
    else:
        train = None
    force_t = dc["t_damage"] if (force_start and arm != "oracle") else None
    res = dm.run_trial(seed, arm, k_eff, incumbent, box, dc, trigger=trigger,
                       duration=duration, train=train, force_trigger_t=force_t)
    row = dm.scalar_metrics(res)
    opt = np.asarray(json.load(open(dm.OPTIMA_JSON))["damaged"]["params"], float)
    d0 = float(np.linalg.norm((incumbent - opt) / rng))
    ap = np.asarray(res["applied"]); kt = int(res["trigger_step"])
    if kt >= 0 and ap.shape[1] > kt:
        d = np.linalg.norm((ap[:, kt:] - opt[:, None]) / rng[:, None], axis=0)
        d_opt_min = float(d.min())
    else:
        d_opt_min = float("nan")
    row.update(arm=arm, scale=float(scale), persist_std=float(persist_std),
               ramp_t=float(ramp_t), force_start=int(bool(force_start)),
               d_opt_start=d0, d_opt_min=d_opt_min)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="cusum")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--ramp", type=float, default=20.0)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--persist-stds", type=float, nargs="+", default=[0.05, 0.10])
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = []
    for s in range(a.seeds):
        # MARX-EFE: forced reaction at ramp start x scale x persist
        for sc, ps in itertools.product(a.scales, a.persist_stds):
            jobs.append((int(s), "marxefe", sc, ps, a.ramp, True, a.trigger,
                         a.duration))
        # references: MARX natural trigger (no force, no persist), noadapt, oracle
        jobs.append((int(s), "marxefe", 0.5, 0.0, a.ramp, False, a.trigger, a.duration))
        jobs.append((int(s), "noadapt", 0.5, 0.0, a.ramp, False, a.trigger, a.duration))
        jobs.append((int(s), "oracle", 0.5, 0.0, a.ramp, False, a.trigger, a.duration))

    print(f"early-adapt sweep: {len(jobs)} runs (trigger={a.trigger}, "
          f"{a.duration:g}s, RR 60->22 Nm, ramp={a.ramp:g}s ending {RAMP_END:g}s)")
    print(f"  MARX forced@ramp-start x scales={a.scales} x persist={a.persist_stds}"
          f"  + refs (marxefe natural, noadapt, oracle)")

    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(a.workers, len(jobs)), maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            if r["arm"] == "marxefe":
                tag = (f"marx s{r['scale']:.1f}/p{r['persist_std']:.2f}"
                       f"{'/F' if r['force_start'] else '/nat'}")
            else:
                tag = r["arm"]
            end = f"FELL(ph{r['fall_phase']})" if r["fell"] else "cap"
            print(f"[{i+1:3d}/{len(jobs)}] {tag:<20} seed{r['seed']:>2} {end:>9} "
                  f"dist={r['dist']:.1f}m vx2={r['mean_vx_ph2']:.2f} "
                  f"d_opt_min={r['d_opt_min']:.2f}", flush=True)

    cols = ["arm", "scale", "persist_std", "force_start", "ramp_t", "seed", "fell",
            "fall_phase", "dist", "t_end", "fall_t", "trigger_t", "n_proposals",
            "mean_vx_ph2", "mean_tip_ph2", "d_opt_start", "d_opt_min"]
    with open(SWEEP_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    def _f(x):
        x = [v for v in x if v == v]
        return float(np.mean(x)) if x else float("nan")

    agg = {}
    for r in rows:
        if r["arm"] == "marxefe" and r["force_start"]:
            tag = f"marx s{r['scale']:.1f} p{r['persist_std']:.2f} forced"
        elif r["arm"] == "marxefe":
            tag = "marx natural (ref)"
        else:
            tag = r["arm"]
        agg.setdefault(tag, []).append(r)
    print(f"\n=== early-adapt summary ({a.seeds} seeds, ramp {a.ramp:g}s) ===")
    print(f"  (d_opt start = {_f([r['d_opt_start'] for r in rows]):.2f}; lower=closer to stable gait)")
    print(f"  {'method':<26} {'falls':>6} {'dist[m]':>8} {'vx2':>6} {'tip2':>6} "
          f"{'d_opt_min':>10} {'nprop':>6}")
    def _key(t):
        return (0 if t == "noadapt" else 2 if t == "oracle"
                else 1 if "natural" in t else 0.5, t)
    for tag in sorted(agg.keys(), key=_key):
        rs = agg[tag]; n = len(rs)
        print(f"  {tag:<26} {sum(int(r['fell']) for r in rs)/n*100:>5.0f}% "
              f"{_f([r['dist'] for r in rs]):>8.1f} "
              f"{_f([r['mean_vx_ph2'] for r in rs]):>6.2f} "
              f"{_f([r['mean_tip_ph2'] for r in rs]):>6.1f} "
              f"{_f([r['d_opt_min'] for r in rs]):>10.2f} "
              f"{_f([r['n_proposals'] for r in rs]):>6.1f}")
    print(f"\nsaved {SWEEP_CSV}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
