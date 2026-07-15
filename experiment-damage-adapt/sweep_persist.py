"""Can the online MARX-EFE agent DISCOVER a recovery gait itself if it keeps
exploring after the damage? (No supplied gait -- pure prediction-error-driven
adaptation.)

Diagnosis (diag): the agent's phase-1 model learns ~zero sensitivity to the
parameters that actually matter post-damage (coupling, b), because they barely
affect a healthy gait; after the trigger it moves the wrong knob (F_FAST) and
never approaches the stable region. It also switches its exploration OFF right
after the trigger.

This sweep turns persistent excitation ON (train["persist"], std=persist_std) so
the agent keeps re-identifying the u->y map under the changed dynamics, and tests
it against the baseline (excitation off post-trigger) for a step (ramp 1 s) and a
slow gradual droop (ramp 20 s, a long SAFE window to explore while only mildly
damaged). Reports fall rate / phase-2 vx and, as a search-progress probe, the
closest normalised approach of the APPLIED parameters to the known stable gait
(damaged_opt) -- the agent is never given this; it is only used to score whether
the search moved toward the stable region.

Usage (from repo root):
    python experiment-damage-adapt/sweep_persist.py --seeds 6 --workers 12
"""

import argparse
import csv
import importlib.util
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
SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_persist.csv")
RAMP_END = 35.0


def _dm():
    spec = importlib.util.spec_from_file_location(
        "dm_p", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _job(args):
    seed, arm, persist_std, ramp_t, trigger, duration = args
    dm = _dm()
    f2s = dm.f2s
    f2s._limit_threads()
    f2s.MARX_CONTROL_PRIOR_SCALE = float(dm.MARX_PRIOR_SCALE)
    f2s.MARX_GOAL_VEL_STD = float(dm.MARX_VEL_STD)
    f2s.MarxEFE.FORGETTING = float(dm.MARX_FORGETTING)
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = os.environ.get("CPG_ATTITUDE_FB", "1") != "0"
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    lo, hi = bl.numpy(), bu.numpy()
    box = (lo, hi)
    rng = hi - lo

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
    res = dm.run_trial(seed, arm, k_eff, incumbent, box, dc, trigger=trigger,
                       duration=duration, train=train)
    row = dm.scalar_metrics(res)

    # search-progress probe: closest normalised approach of APPLIED params to the
    # known stable gait over the post-trigger trajectory (diagnostic only).
    opt = json.load(open(dm.OPTIMA_JSON))["damaged"]["params"]
    opt = np.asarray(opt, float)
    d0 = float(np.linalg.norm((incumbent - opt) / rng))
    ap = np.asarray(res["applied"])
    kt = int(res["trigger_step"])
    if kt >= 0 and ap.shape[1] > kt:
        seg = ap[:, kt:]
        d = np.linalg.norm((seg - opt[:, None]) / rng[:, None], axis=0)
        d_opt_min = float(d.min())
    else:
        d_opt_min = float("nan")
    row.update(arm=arm, persist_std=float(persist_std), ramp_t=float(ramp_t),
               d_opt_start=d0, d_opt_min=d_opt_min)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="cusum")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--persist-stds", type=float, nargs="+", default=[0.0, 0.05, 0.10],
                    help="post-trigger excitation std (frac of range); 0 = off (baseline)")
    ap.add_argument("--ramps", type=float, nargs="+", default=[1, 20])
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = []
    for s in range(a.seeds):
        for rt in a.ramps:
            for ps in a.persist_stds:
                jobs.append((int(s), "marxefe", ps, rt, a.trigger, a.duration))
            # references (persist n/a) per ramp:
            for arm in ("noadapt", "oracle"):
                jobs.append((int(s), arm, 0.0, rt, a.trigger, a.duration))

    print(f"persistent-excitation sweep: {len(jobs)} runs (trigger={a.trigger}, "
          f"{a.duration:g}s, RR 60->22 Nm, ramp-end={RAMP_END:g}s)")
    print(f"  marxefe persist_stds={a.persist_stds}  ramps={a.ramps}  "
          f"+ noadapt/oracle refs")

    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(a.workers, len(jobs)), maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            tag = (f"marxefe/p{r['persist_std']:.2f}" if r["arm"] == "marxefe"
                   else r["arm"])
            end = f"FELL(ph{r['fall_phase']})" if r["fell"] else "cap"
            print(f"[{i+1:3d}/{len(jobs)}] ramp={r['ramp_t']:>4.0f}s {tag:<14} "
                  f"seed{r['seed']:>2} {end:>9} dist={r['dist']:.1f}m "
                  f"vx2={r['mean_vx_ph2']:.2f} d_opt_min={r['d_opt_min']:.2f}",
                  flush=True)

    cols = ["arm", "persist_std", "ramp_t", "seed", "fell", "fall_phase", "dist",
            "t_end", "fall_t", "trigger_t", "n_proposals", "mean_vx_ph2",
            "mean_tip_ph2", "d_opt_start", "d_opt_min"]
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
        tag = (f"marxefe p={r['persist_std']:.2f}" if r["arm"] == "marxefe"
               else r["arm"])
        agg.setdefault((r["ramp_t"], tag), []).append(r)
    print(f"\n=== persistent-excitation summary ({a.seeds} seeds) ===")
    print(f"  (d_opt: normalised param distance to the stable gait; "
          f"start={_f([r['d_opt_start'] for r in rows]):.2f}, lower=closer)")
    print(f"  {'ramp[s]':>7} {'method':<16} {'falls':>6} {'dist[m]':>8} {'vx2':>6} "
          f"{'tip2':>6} {'d_opt_min':>10} {'nprop':>6}")
    for key in sorted(agg.keys(), key=lambda k: (k[0], k[1])):
        rt, tag = key
        rs = agg[key]
        n = len(rs)
        print(f"  {rt:>7.0f} {tag:<16} "
              f"{sum(int(r['fell']) for r in rs)/n*100:>5.0f}% "
              f"{_f([r['dist'] for r in rs]):>8.1f} "
              f"{_f([r['mean_vx_ph2'] for r in rs]):>6.2f} "
              f"{_f([r['mean_tip_ph2'] for r in rs]):>6.1f} "
              f"{_f([r['d_opt_min'] for r in rs]):>10.2f} "
              f"{_f([r['n_proposals'] for r in rs]):>6.1f}")
    print(f"\nsaved {SWEEP_CSV}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
