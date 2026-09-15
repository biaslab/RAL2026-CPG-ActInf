"""Chase the LATENCY lever for the leg-damage experiment.

The MARX-EFE hyper-parameter sweep showed every config falls 100%; detection
latency was a flat ~1.8 s and the adapters got only a few proposals before the
robot tipped. This script decomposes detection latency from online-search
capability by running the searching arms under three trigger conditions:

  * natural   -- the normal cusum monitor (h=5.0), i.e. run_experiment.py;
  * sensitive -- a more sensitive cusum (lower h / slack) that fires sooner;
  * forced    -- trigger FORCED at the damage onset (zero detection latency): the
                 method gets the oracle's TIMING but must still search for the
                 params online. If it survives here but not under `natural`,
                 latency is the lever; if it still falls, the bottleneck is the
                 online search, not detection.

noadapt (falls) and oracle (zero-latency + known params, survives) are included
as the lower/upper reference under the natural trigger. MARX-EFE uses the best
config from sweep_marxefe.py (wide control prior); override with --marx-*.

Usage (from repo root):
    python experiment-damage-adapt/sweep_latency.py --seeds 8 --workers 12
"""

import argparse
import csv
import importlib.util
import os
import sys
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_HERE, "results")
SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_latency.csv")

# Trigger conditions: (name, cusum_h, cusum_slack, force_at_damage)
CONDITIONS = [
    ("natural",   5.0, 0.10, False),
    ("sensitive", 1.5, 0.05, False),
    ("forced",    5.0, 0.10, True),
]
SEARCH_ARMS = ["marxefe", "bo", "grid"]
REF_ARMS = ["noadapt", "oracle"]        # run under `natural` only, for reference


def _dm():
    spec = importlib.util.spec_from_file_location(
        "dm_lat", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_marxefe(f2s, input_buffer):
    base = f2s.MarxEFE
    ib = int(input_buffer)

    class MarxEFEIB(base):
        def __init__(self, incumbent, oracle_params, box, seed):
            from methods.marxefe_optimizer import build_marx_agent
            np.random.seed(1)
            goal_std = (f2s.MARX_GOAL_VEL_STD, f2s.MARX_GOAL_VEL_STD,
                        np.deg2rad(f2s.MON_PITCH_DEG), np.deg2rad(f2s.MON_ROLL_DEG))
            self.agent = build_marx_agent(
                target_velocity=f2s.TARGET_VX,
                control_prior_scale=f2s.MARX_CONTROL_PRIOR_SCALE,
                goal_prior_std=goal_std, time_horizon=2,
                forgetting=self.FORGETTING, input_buffer=ib)
            self.agent.μ = np.asarray(incumbent, float).copy()
            lo, hi = box
            self.lims = [(float(lo[i]), float(hi[i]))
                         for _ in range(self.agent.thorizon) for i in range(8)]
            self.lo, self.hi = lo, hi
            self.inc = incumbent

    return MarxEFEIB


def _job(args):
    (seed, arm, cond_name, cusum_h, cusum_slack, force, trigger, duration,
     marx_scale, marx_vel_std, marx_delay) = args
    dm = _dm()
    f2s = dm.f2s
    f2s._limit_threads()
    f2s.DT_CUSUM_H = float(cusum_h)
    f2s.DT_CUSUM_SLACK = float(cusum_slack)
    f2s.MARX_CONTROL_PRIOR_SCALE = float(marx_scale)
    f2s.MARX_GOAL_VEL_STD = float(marx_vel_std)
    f2s.MarxEFE.FORGETTING = float(dm.MARX_FORGETTING)
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = os.environ.get("CPG_ATTITUDE_FB", "1") != "0"
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    box = (bl.numpy(), bu.numpy())
    if arm == "marxefe":
        dm.METHODS["marxefe"] = _make_marxefe(f2s, marx_delay + 1)

    k_eff = 1.0 if trigger in ("dt", "cusum") else float(f2s.K_DEFAULT)
    incumbent = dm.load_incumbent()
    dc = dm.damage_defaults(duration)
    train = (dict(t0=float(dm.TRAIN_T0), std=float(dm.TRAIN_STD),
                  tau=float(dm.TRAIN_TAU)) if arm == "marxefe" else None)
    force_t = dc["t_damage"] if force else None
    res = dm.run_trial(seed, arm, k_eff, incumbent, box, dc,
                       trigger=trigger, duration=duration, train=train,
                       force_trigger_t=force_t)
    row = dm.scalar_metrics(res)
    t_dmg = float(res["t_shift"])
    # effective trigger latency = when adaptation actually started - damage onset
    row["eff_lat"] = (row["trigger_t"] - t_dmg
                      if row["triggered"] else float("nan"))
    row.update(arm=arm, cond=cond_name, marx_scale=marx_scale,
               marx_vel_std=marx_vel_std, marx_delay=marx_delay)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="cusum")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--marx-scale", type=float, default=0.8,
                    help="MARX control-prior scale (sweep best: wide=0.8)")
    ap.add_argument("--marx-vel-std", type=float, default=0.2)
    ap.add_argument("--marx-delay", type=int, default=2,
                    help="MARX input-buffer delay_inp")
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = []
    for s in range(a.seeds):
        for (cname, ch, cs, force) in CONDITIONS:
            for arm in SEARCH_ARMS:
                jobs.append((int(s), arm, cname, ch, cs, force, a.trigger,
                             a.duration, a.marx_scale, a.marx_vel_std, a.marx_delay))
        # references only under the natural trigger
        for arm in REF_ARMS:
            jobs.append((int(s), arm, "natural", 5.0, 0.10, False, a.trigger,
                         a.duration, a.marx_scale, a.marx_vel_std, a.marx_delay))

    print(f"latency sweep: {len(jobs)} runs (trigger={a.trigger}, {a.duration:g}s, "
          f"RR 60->22 Nm; MARX scale={a.marx_scale} velstd={a.marx_vel_std} "
          f"delay={a.marx_delay})")
    print(f"  conditions: {[c[0] for c in CONDITIONS]}  search arms: {SEARCH_ARMS}")

    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(a.workers, len(jobs)), maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            end = f"FELL(ph{r['fall_phase']})" if r["fell"] else "cap"
            print(f"[{i+1:3d}/{len(jobs)}] {r['arm']:<8} {r['cond']:<9} "
                  f"seed{r['seed']:>2} {end:>9} dist={r['dist']:.1f}m "
                  f"vx2={r['mean_vx_ph2']:.2f} efflat={r['eff_lat']:.1f}s "
                  f"nprop={r['n_proposals']}", flush=True)

    cols = ["arm", "cond", "seed", "fell", "fall_phase", "dist", "t_end",
            "trigger_t", "eff_lat", "det_latency", "n_triggers", "n_proposals",
            "mean_vx_ph2", "mean_tip_ph2", "power_ph2",
            "marx_scale", "marx_vel_std", "marx_delay"]
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
        agg.setdefault((r["arm"], r["cond"]), []).append(r)
    print(f"\n=== latency sweep summary ({a.seeds} seeds) ===")
    print(f"  {'arm':<8} {'cond':<9} {'falls':>6} {'dist[m]':>8} {'vx2':>6} "
          f"{'tip2':>6} {'eff_lat[s]':>10} {'nprop':>6}")
    order = {("noadapt", "natural"): 0}
    def _sortkey(k):
        arm, cond = k
        ai = SEARCH_ARMS.index(arm) + 1 if arm in SEARCH_ARMS else (
            0 if arm == "noadapt" else 99)
        ci = [c[0] for c in CONDITIONS].index(cond) if cond in [c[0] for c in CONDITIONS] else 0
        return (ai, ci)
    for key in sorted(agg.keys(), key=_sortkey):
        rs = agg[key]
        n = len(rs)
        arm, cond = key
        print(f"  {arm:<8} {cond:<9} "
              f"{sum(int(r['fell']) for r in rs)/n*100:>5.0f}% "
              f"{_f([r['dist'] for r in rs]):>8.1f} "
              f"{_f([r['mean_vx_ph2'] for r in rs]):>6.2f} "
              f"{_f([r['mean_tip_ph2'] for r in rs]):>6.1f} "
              f"{_f([r['eff_lat'] for r in rs]):>10.2f} "
              f"{_f([r['n_proposals'] for r in rs]):>6.1f}")
    print(f"\nsaved {SWEEP_CSV}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
