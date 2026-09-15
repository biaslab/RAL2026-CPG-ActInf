"""Does GRADUAL leg damage (a slow torque droop) give the online methods enough
time to adapt? Sweeps the damage ramp duration `ramp_t` from the ~1 s step of
run_experiment.py up to a slow multi-second drift.

Confound control: a longer ramp at a fixed START time would leave less time at
full damage within the bout, so survivors might just get less full-damage
exposure. Instead the ramp END is fixed (--ramp-end, default 35 s) and the start
slides earlier for longer ramps, so every setting spends the SAME time at the
final 22 Nm (ramp_end .. duration). The only thing that varies is how gradual the
descent to 22 Nm is.

For each ramp_t x arm x seed it runs one bout with the standard protocol (cusum
trigger, RR 60->22 Nm) and logs the usual metrics; a per-(arm, ramp) summary is
printed. noadapt / oracle bound the range. MARX-EFE uses the run_experiment
defaults (control prior 0.5, vel std 0.2, input delay 2).

Usage (from repo root):
    python experiment-damage-adapt/sweep_gradual.py --seeds 6 --workers 12
    python experiment-damage-adapt/sweep_gradual.py --ramps 1 5 10 20 --ramp-end 35
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
SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_gradual.csv")


def _dm():
    spec = importlib.util.spec_from_file_location(
        "dm_grad", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _job(args):
    seed, arm, ramp_t, ramp_end, trigger, duration = args
    dm = _dm()
    f2s = dm.f2s
    f2s._limit_threads()
    # MARX knobs = run_experiment defaults (the MarxEFE ctor reads these globals):
    f2s.MARX_CONTROL_PRIOR_SCALE = float(dm.MARX_PRIOR_SCALE)
    f2s.MARX_GOAL_VEL_STD = float(dm.MARX_VEL_STD)
    f2s.MarxEFE.FORGETTING = float(dm.MARX_FORGETTING)
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = os.environ.get("CPG_ATTITUDE_FB", "1") != "0"
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    box = (bl.numpy(), bu.numpy())

    k_eff = 1.0 if trigger in ("dt", "cusum") else float(f2s.K_DEFAULT)
    incumbent = dm.load_incumbent()
    dc = dm.damage_defaults(duration)
    dc["ramp_t"] = float(ramp_t)
    dc["t_damage"] = float(ramp_end) - float(ramp_t)      # fixed ramp END
    train = (dict(t0=float(dm.TRAIN_T0), std=float(dm.TRAIN_STD),
                  tau=float(dm.TRAIN_TAU)) if arm == "marxefe" else None)
    res = dm.run_trial(seed, arm, k_eff, incumbent, box, dc,
                       trigger=trigger, duration=duration, train=train)
    row = dm.scalar_metrics(res)
    t_dmg = float(res["t_shift"])                         # ramp start
    row["eff_lat"] = (row["trigger_t"] - t_dmg if row["triggered"]
                      else float("nan"))
    # fall time relative to the (fixed) ramp end: <0 fell during ramp, >0 after
    row["fall_rel_end"] = (row["fall_t"] - float(ramp_end)
                           if row["fell"] else float("nan"))
    row.update(arm=arm, ramp_t=float(ramp_t), t_damage=dc["t_damage"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="cusum")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--ramps", type=float, nargs="+", default=[1, 5, 10, 20],
                    help="damage ramp durations [s] (1 ~ the run_experiment step)")
    ap.add_argument("--ramp-end", type=float, default=35.0,
                    help="fixed time [s] the leg reaches full damage; the ramp "
                         "starts ramp_t before this so full-damage exposure "
                         "(ramp_end..duration) is constant across ramps")
    ap.add_argument("--arms", nargs="+",
                    default=["noadapt", "grid", "bo", "marxefe", "oracle"],
                    choices=["noadapt", "grid", "bo", "marxefe", "oracle"])
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = [(int(s), arm, rt, a.ramp_end, a.trigger, a.duration)
            for rt in a.ramps for arm in a.arms for s in range(a.seeds)]
    print(f"gradual-damage sweep: {len(a.ramps)} ramps x {len(a.arms)} arms x "
          f"{a.seeds} seeds = {len(jobs)} runs (trigger={a.trigger}, "
          f"{a.duration:g}s, RR 60->22 Nm, ramp-end={a.ramp_end:g}s)")
    print(f"  ramps={a.ramps}  arms={a.arms}")

    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(a.workers, len(jobs)), maxtasksperchild=2) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            end = f"FELL(ph{r['fall_phase']})" if r["fell"] else "cap"
            print(f"[{i+1:3d}/{len(jobs)}] ramp={r['ramp_t']:>4.0f}s {r['arm']:<8} "
                  f"seed{r['seed']:>2} {end:>9} dist={r['dist']:.1f}m "
                  f"vx2={r['mean_vx_ph2']:.2f} lat={r['eff_lat']:.1f}s "
                  f"nprop={r['n_proposals']}", flush=True)

    cols = ["ramp_t", "arm", "seed", "t_damage", "fell", "fall_phase", "fall_t",
            "fall_rel_end", "dist", "t_end", "trigger_t", "eff_lat", "n_triggers",
            "n_proposals", "mean_vx_ph1", "mean_vx_ph2", "mean_tip_ph2", "power_ph2"]
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
        agg.setdefault((r["arm"], r["ramp_t"]), []).append(r)
    print(f"\n=== gradual-damage summary ({a.seeds} seeds), by arm x ramp ===")
    print(f"  {'arm':<8} {'ramp[s]':>7} {'falls':>6} {'dist[m]':>8} {'vx2':>6} "
          f"{'tip2':>6} {'eff_lat[s]':>10} {'nprop':>6} {'fall_vs_end[s]':>14}")
    for arm in a.arms:
        for rt in a.ramps:
            rs = agg.get((arm, float(rt)))
            if not rs:
                continue
            n = len(rs)
            print(f"  {arm:<8} {rt:>7.0f} "
                  f"{sum(int(r['fell']) for r in rs)/n*100:>5.0f}% "
                  f"{_f([r['dist'] for r in rs]):>8.1f} "
                  f"{_f([r['mean_vx_ph2'] for r in rs]):>6.2f} "
                  f"{_f([r['mean_tip_ph2'] for r in rs]):>6.1f} "
                  f"{_f([r['eff_lat'] for r in rs]):>10.2f} "
                  f"{_f([r['n_proposals'] for r in rs]):>6.1f} "
                  f"{_f([r['fall_rel_end'] for r in rs]):>14.1f}")
        print()
    print(f"saved {SWEEP_CSV}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
