"""Grid sweep of the MARX-EFE hyper-parameters for the leg-damage experiment.

Sweeps three knobs of the active-inference agent, holding the rest of the
protocol identical to `run_experiment.py` (cusum trigger, 60 s bout, RR leg
60->22 Nm, phase-1 OU training on):

  * control-prior scale  (--prior-scales)  -- MARX_CONTROL_PRIOR_SCALE; the EFE
        trust region, sigma = scale * range / (2*n_sigma). Small pins to the
        incumbent, large lets the EFE jump (and overshoot to the box bounds).
  * goal-prior vel std   (--vel-stds)       -- goal_prior_std on (vx, vy) [m/s];
        tight = strong velocity-tracking pressure, loose = tolerates the deficit.
  * input buffer         (--input-delays)   -- delay_inp of the MARX AR model
        (ubuffer width = delay_inp + 1). 0 = only the current u (memoryless in
        the input), 1/2 = one/two past inputs; the run_experiment default is 2.

For each (scale, vel_std, input_delay) x seed it runs one bout and logs the same
scalar metrics as the main experiment. Per-run rows go to
`results/sweep_marxefe.csv`; a ranked per-config summary (fall rate, phase-2
velocity, survival distance, tip, detection latency) is printed best-first.

Usage (from repo root):
    python experiment-damage-adapt/sweep_marxefe.py --seeds 6 --workers 10
    python experiment-damage-adapt/sweep_marxefe.py \
        --prior-scales 0.25 0.5 0.8 --vel-stds 0.1 0.2 0.4 --input-delays 0 1 2
"""

import argparse
import csv
import importlib.util
import itertools
import os
import sys
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_HERE, "results")
SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_marxefe.csv")

# run_experiment.py defaults (the incumbent config), for the "*" marker.
DEF_SCALE, DEF_VEL_STD, DEF_INPUT_DELAY = 0.5, 0.2, 2


def _dm():
    """Load experiment-damage-adapt/run_experiment.py as a fresh module."""
    spec = importlib.util.spec_from_file_location(
        "dm_sweep", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_marxefe(f2s, input_buffer):
    """MarxEFE subclass that builds its agent with a chosen input_buffer
    (delay_inp = input_buffer - 1). Identical to f2s.MarxEFE otherwise; reads the
    control-prior / goal-prior globals set on f2s at construction time."""
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
    seed, scale, vel_std, input_delay, trigger, duration = args
    dm = _dm()
    f2s = dm.f2s
    f2s._limit_threads()
    # MARX knobs (read by the MarxEFE constructor via f2s globals):
    f2s.MARX_CONTROL_PRIOR_SCALE = float(scale)
    f2s.MARX_GOAL_VEL_STD = float(vel_std)
    f2s.MarxEFE.FORGETTING = float(dm.MARX_FORGETTING)
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = os.environ.get("CPG_ATTITUDE_FB", "1") != "0"
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    box = (bl.numpy(), bu.numpy())
    # inject the input-buffer-tuned MarxEFE into the damage module's METHODS:
    dm.METHODS["marxefe"] = _make_marxefe(f2s, input_delay + 1)

    k_eff = 1.0 if trigger in ("dt", "cusum") else float(f2s.K_DEFAULT)
    incumbent = dm.load_incumbent()
    dc = dm.damage_defaults(duration)
    train = dict(t0=float(dm.TRAIN_T0), std=float(dm.TRAIN_STD),
                 tau=float(dm.TRAIN_TAU))
    res = dm.run_trial(seed, "marxefe", k_eff, incumbent, box, dc,
                       trigger=trigger, duration=duration, train=train)
    row = dm.scalar_metrics(res)
    row.update(prior_scale=float(scale), vel_std=float(vel_std),
               input_delay=int(input_delay))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="cusum")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--prior-scales", type=float, nargs="+",
                    default=[0.25, 0.5, 0.8])
    ap.add_argument("--vel-stds", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    ap.add_argument("--input-delays", type=int, nargs="+", default=[0, 1, 2],
                    help="delay_inp values (ubuffer width = delay+1); default's 2")
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    grid = list(itertools.product(a.prior_scales, a.vel_stds, a.input_delays))
    jobs = [(int(s), sc, vs, idl, a.trigger, a.duration)
            for (sc, vs, idl) in grid for s in range(a.seeds)]
    print(f"MARX-EFE sweep: {len(grid)} configs x {a.seeds} seeds = {len(jobs)} "
          f"runs (trigger={a.trigger}, {a.duration:g}s bouts, RR 60->22 Nm)")
    print(f"  prior_scales={a.prior_scales}  vel_stds={a.vel_stds}  "
          f"input_delays={a.input_delays}")

    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(a.workers, len(jobs)), maxtasksperchild=2) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(row)
            end = f"FELL(ph{row['fall_phase']})" if row["fell"] else "cap"
            print(f"[{i+1:3d}/{len(jobs)}] s={row['prior_scale']:.2f} "
                  f"vstd={row['vel_std']:.2f} in={row['input_delay']} "
                  f"seed{row['seed']:>2} {end:>9} dist={row['dist']:.1f}m "
                  f"vx2={row['mean_vx_ph2']:.2f} lat={row['det_latency']:.1f}s "
                  f"nprop={row['n_proposals']}", flush=True)

    # per-run CSV
    cols = ["prior_scale", "vel_std", "input_delay", "seed", "fell", "fall_phase",
            "dist", "t_end", "det_latency", "n_triggers", "n_proposals",
            "mean_vx_ph1", "mean_vx_ph2", "mean_tip_ph1", "mean_tip_ph2",
            "power_ph2", "mean_J"]
    with open(SWEEP_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # per-config aggregate, ranked (fewest falls, then fastest phase-2)
    def _f(x):
        x = [v for v in x if v == v]
        return float(np.mean(x)) if x else float("nan")

    agg = {}
    for r in rows:
        key = (r["prior_scale"], r["vel_std"], r["input_delay"])
        agg.setdefault(key, []).append(r)
    summary = []
    for key, rs in agg.items():
        n = len(rs)
        summary.append(dict(
            key=key, n=n,
            fall_rate=sum(int(r["fell"]) for r in rs) / n,
            dist=_f([r["dist"] for r in rs]),
            vx2=_f([r["mean_vx_ph2"] for r in rs]),
            tip2=_f([r["mean_tip_ph2"] for r in rs]),
            lat=_f([r["det_latency"] for r in rs]),
            nprop=_f([r["n_proposals"] for r in rs])))
    summary.sort(key=lambda d: (d["fall_rate"], -d["vx2"]))

    print(f"\n=== MARX-EFE sweep summary ({a.seeds} seeds/config), "
          f"ranked by fall rate then phase-2 vx ===")
    print(f"  {'scale':>6} {'velstd':>6} {'in':>3}   {'falls':>6} {'dist[m]':>8} "
          f"{'vx2':>6} {'tip2':>6} {'lat[s]':>7} {'nprop':>6}")
    for d in summary:
        sc, vs, idl = d["key"]
        star = " *" if (sc == DEF_SCALE and vs == DEF_VEL_STD
                        and idl == DEF_INPUT_DELAY) else "  "
        print(f"{star}{sc:>6.2f} {vs:>6.2f} {idl:>3d}   "
              f"{d['fall_rate']*100:>5.0f}% {d['dist']:>8.1f} {d['vx2']:>6.2f} "
              f"{d['tip2']:>6.1f} {d['lat']:>7.1f} {d['nprop']:>6.1f}")
    print("  (* = run_experiment.py default config)")
    print(f"\nsaved {SWEEP_CSV}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
