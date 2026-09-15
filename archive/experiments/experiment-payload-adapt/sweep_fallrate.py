"""Sweep the payload-shift scenario parameters for a HIGH no-adapt fall rate.

Motivation (see README regime map): at the current default (8 kg, 0.20 m
lat+back) no-adapt merely limps, at 0.25 m it always falls but NO gait
survives (ice-like, adaptation cannot pay off either), and 10 kg / 0.30 m is a
pure transition shock. This script scans the region in between along four
axes -- payload mass, offset magnitude, offset direction (lateral- vs
rear-heavy), and payload height (lever arm) -- to find a setting where

  (a) the incumbent (flat-optimal) gait falls in MOST seeds after the shift,
  (b) the falls are persistent-mismatch falls, not ramp shocks
      (fall time >= SHOCK_T after shift start), and
  (c) the condition is RECOVERABLE: some forward-walking gait survives it,
      so a clairvoyant/adaptive parameter switch can in principle help.

Stages (from repo root, Anaconda base python):
    python experiment-payload-adapt/sweep_fallrate.py sweep   [--seeds 10 --workers 10]
    python experiment-payload-adapt/sweep_fallrate.py screen  [--gaits 12 --seeds 2]
    python experiment-payload-adapt/sweep_fallrate.py report

`sweep` measures the no-adapt fall rate per setting (30 s bouts, shift at
15 s); `screen` takes every setting with fall rate >= --min-fall-rate and
evaluates LHS-sampled gaits under the always-shifted condition (shift at 1 s
while walking, as in fit_payload_oracles.py); `report` prints the merged
table. Results accumulate in results/sweep_fallrate.csv / sweep_screen.csv.
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

RESULTS = os.path.join(_HERE, "results")
SWEEP_CSV = os.path.join(RESULTS, "sweep_fallrate.csv")
SCREEN_CSV = os.path.join(RESULTS, "sweep_screen.csv")

SWEEP_DURATION = 30.0     # bout [s]; shift at 15 s (falls occur 0.7-5 s post-shift)
SCREEN_DURATION = 15.0    # always-shifted screen bout [s] (shift at 1 s, walking)
SHOCK_T = 1.5             # fall sooner than this after shift start = ramp shock [s]
FWD_VX = 0.10             # a "forward-walking survivor" must average this [m/s]

# (name, mass [kg], lat [m], back [m], up [m]) -- default up is 0.15.
# A/D anchor the known regime map (0 falls / 3-3 falls-unrecoverable).
SETTINGS = [
    ("A_8kg_0.20",        8.0, 0.200, 0.200, 0.15),
    ("B_8kg_0.225",       8.0, 0.225, 0.225, 0.15),
    ("C_8kg_0.2375",      8.0, 0.2375, 0.2375, 0.15),
    ("D_8kg_0.25",        8.0, 0.250, 0.250, 0.15),
    ("E_9kg_0.20",        9.0, 0.200, 0.200, 0.15),
    ("F_9kg_0.225",       9.0, 0.225, 0.225, 0.15),
    ("G_10kg_0.20",      10.0, 0.200, 0.200, 0.15),
    ("H_10kg_0.225",     10.0, 0.225, 0.225, 0.15),
    ("I_8kg_lat0.30",     8.0, 0.300, 0.100, 0.15),
    ("J_8kg_back0.30",    8.0, 0.100, 0.300, 0.15),
    ("K_8kg_0.20_up0.30", 8.0, 0.200, 0.200, 0.30),
    ("L_8kg_0.23",        8.0, 0.230, 0.230, 0.15),  # cliff probe: B recoverable, C not
]
BY_NAME = {s[0]: s for s in SETTINGS}


def _pl():
    spec = importlib.util.spec_from_file_location(
        "pl_sweep", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cfg(pl, mass, lat, back, up, t_shift, duration):
    pc = pl.payload_defaults(duration)
    pc.update(mass=float(mass), lat=float(lat), back=float(back), up=float(up),
              t_shift=t_shift)
    return pc


def _worker_env():
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = True
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from methods.cpg_bounds import bounds_lower, bounds_upper
    return bounds_lower.numpy(), bounds_upper.numpy()


# ── Stage 1: no-adapt fall rate over the settings grid ───────────────────────

def _sweep_job(args):
    name, seed, duration = args
    lo, hi = _worker_env()
    pl = _pl()
    _, mass, lat, back, up = BY_NAME[name]
    incumbent = pl.load_incumbent()
    pc = _cfg(pl, mass, lat, back, up, duration / 2.0, duration)
    res = pl.run_trial(seed, "noadapt", 1.0, incumbent, (lo, hi), pc,
                       trigger="cusum", duration=duration)
    row = pl.scalar_metrics(res)
    t_shift = duration / 2.0
    fall_after = (row["fall_t"] - t_shift) if row["fell"] else np.nan
    return dict(setting=name, seed=seed, mass=mass, lat=lat, back=back, up=up,
                duration=duration, fell=row["fell"],
                fall_after_shift=fall_after,
                shock=int(row["fell"] and fall_after < SHOCK_T),
                dist=row["dist"], det_latency=row["det_latency"],
                mean_vx_ph1=row["mean_vx_ph1"], mean_vx_ph2=row["mean_vx_ph2"],
                mean_tip_ph2=row["mean_tip_ph2"], power_ph2=row["power_ph2"])


def stage_sweep(seeds, workers, duration, only=None):
    names = [n for n in BY_NAME if (only is None or n in only)]
    done = set()
    rows = []
    if os.path.exists(SWEEP_CSV):
        with open(SWEEP_CSV) as f:
            rows = list(csv.DictReader(f))
        done = {(r["setting"], int(r["seed"])) for r in rows}
    jobs = [(n, s, duration) for n in names for s in range(seeds)
            if (n, s) not in done]
    print(f"sweep: {len(names)} settings x {seeds} seeds -> {len(jobs)} new trials "
          f"({len(done)} already in {os.path.basename(SWEEP_CSV)})", flush=True)
    if jobs:
        ctx = get_context("spawn")
        with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=4) as pool:
            for i, row in enumerate(pool.imap_unordered(_sweep_job, jobs)):
                rows.append({k: str(v) for k, v in row.items()})
                fa = row["fall_after_shift"]
                status = ("FELL %+5.1fs%s" % (fa, " SHOCK" if row["shock"] else "")
                          if row["fell"] else "survived")
                print(f"[{i+1:3d}/{len(jobs)}] {row['setting']:<18} seed{row['seed']:>2} "
                      f"{status:<16} vx2={row['mean_vx_ph2']:.2f}", flush=True)
        os.makedirs(RESULTS, exist_ok=True)
        cols = list(rows[0].keys())
        with open(SWEEP_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"saved {SWEEP_CSV} ({len(rows)} rows)")
    report()


# ── Stage 2: recoverability screen (LHS gaits under always-shifted) ──────────

def _screen_job(args):
    name, gait_id, params, seed, duration = args
    lo, hi = _worker_env()
    pl = _pl()
    _, mass, lat, back, up = BY_NAME[name]
    pc = _cfg(pl, mass, lat, back, up, 1.0, duration)  # shift at 1 s, walking
    res = pl.run_trial(seed, "noadapt", 1.0, np.asarray(params, float),
                       (lo, hi), pc, trigger="cusum", duration=duration)
    tail = np.asarray(res["t"]) >= 5.0
    vx = float(np.mean(np.asarray(res["vx"])[tail])) if tail.any() else np.nan
    tip = float(np.mean(pl._tip_dev_deg(res["roll"], res["pitch"])[tail])) \
        if tail.any() else np.nan
    return dict(setting=name, gait=gait_id, seed=seed, fell=int(res["fell"]),
                mean_vx_tail=vx, mean_tip_tail=tip,
                params=" ".join(f"{v:.4f}" for v in np.asarray(params).ravel()))


def stage_screen(n_gaits, seeds, workers, min_fall_rate, only=None):
    summ = _sweep_summary()
    names = [n for n, s in summ.items()
             if (only and n in only) or (not only and s["fall_rate"] >= min_fall_rate)]
    if not names:
        print(f"no settings with fall rate >= {min_fall_rate}; nothing to screen")
        return
    pl = _pl()
    incumbent = pl.load_incumbent()
    from methods.cpg_bounds import bounds_lower, bounds_upper
    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    from scipy.stats import qmc
    lhs = qmc.LatinHypercube(d=len(lo), seed=7)
    gaits = {"incumbent": incumbent}
    for i, u in enumerate(lhs.random(n_gaits)):
        gaits[f"lhs{i}"] = lo + u * (hi - lo)

    done, rows = set(), []
    if os.path.exists(SCREEN_CSV):
        with open(SCREEN_CSV) as f:
            rows = list(csv.DictReader(f))
        done = {(r["setting"], r["gait"], int(r["seed"])) for r in rows}
    jobs = [(n, g, p, s, SCREEN_DURATION)
            for n in names for g, p in gaits.items() for s in range(seeds)
            if (n, g, s) not in done]
    print(f"screen: {names} x {len(gaits)} gaits x {seeds} seeds -> "
          f"{len(jobs)} new trials", flush=True)
    if jobs:
        ctx = get_context("spawn")
        with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=4) as pool:
            for i, row in enumerate(pool.imap_unordered(_screen_job, jobs)):
                rows.append({k: str(v) for k, v in row.items()})
                print(f"[{i+1:3d}/{len(jobs)}] {row['setting']:<18} "
                      f"{row['gait']:<10} seed{row['seed']} "
                      f"{'FELL' if row['fell'] else 'ok':<5} "
                      f"vx={row['mean_vx_tail']:.2f}", flush=True)
        cols = list(rows[0].keys())
        with open(SCREEN_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"saved {SCREEN_CSV} ({len(rows)} rows)")
    report()


# ── Stage 3: BO recoverability check (targeted, not random) ─────────────────
# The LHS screen samples the 8-D box too sparsely to prove a setting
# unrecoverable; this fits a gait to the always-shifted condition with GP-UCB
# (same recipe as fit_payload_oracles.py) and reports the best found.

BO_CSV = os.path.join(RESULTS, "sweep_bo_recover.csv")


def _bo_score(pl, box, name, params, seeds, duration):
    """Mean V (fit_payload_oracles-style tail score) + tail vx, always-shifted."""
    _, mass, lat, back, up = BY_NAME[name]
    vs, vxs, fell = [], [], 0
    for s in range(seeds):
        pc = _cfg(pl, mass, lat, back, up, 1.0, duration)
        res = pl.run_trial(s, "noadapt", 1.0, np.asarray(params, float), box,
                           pc, trigger="cusum", duration=duration)
        t = np.asarray(res["t"])
        tail = t >= 5.0
        if int(res["fell"]) or tail.sum() < 200:
            vs.append(-2.0)
            fell += 1
            continue
        vx = np.asarray(res["vx"])[tail]
        roll = np.rad2deg(np.asarray(res["roll"])[tail])
        pitch = np.rad2deg(np.asarray(res["pitch"])[tail])
        r_v = min(max(float(np.mean(vx)), 0.0) / 0.5, 1.0)
        vs.append(r_v - (np.sqrt(np.mean(roll ** 2))
                         + np.sqrt(np.mean((pitch - np.median(pitch)) ** 2))) / 10.0)
        vxs.append(float(np.mean(vx)))
    return float(np.mean(vs)), (float(np.mean(vxs)) if vxs else np.nan), fell


def _bo_job(args):
    name, n_trials, seeds, duration = args
    lo, hi = _worker_env()
    box = (lo, hi)
    import torch
    from methods.bo_optimizer import BOOptimizer, BetaSchedule
    pl = _pl()
    incumbent = pl.load_incumbent()
    bo = BOOptimizer(
        bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
        beta_schedule=BetaSchedule(beta_init=5.0, beta_min=1.0,
                                   n_decay_start=max(10, n_trials // 2), gamma=0.9),
        n_init=8, seed=hash(name) % 9973)
    rng = np.random.default_rng(hash(name) % 9973)
    best = None
    for t in range(n_trials):
        if t == 0:
            x = incumbent.copy()
        elif t <= 8:
            x = rng.uniform(lo, hi)
        else:
            try:
                model = bo.fit_model()
                x = bo.from_unit(bo.suggest(model, bo.beta_schedule(t)))
            except Exception:
                x = rng.uniform(lo, hi)
        x = np.clip(np.asarray(x, float), lo, hi)
        V, vx, fell = _bo_score(pl, box, name, x, seeds, duration)
        bo._append(x, V)
        if best is None or V > best[0]:
            best = (V, vx, fell, x.copy())
        print(f"  [{name}] trial {t+1:2d}/{n_trials} V={V:+.3f} vx={vx:.2f} "
              f"fell={fell}/{seeds} | best V={best[0]:+.3f}", flush=True)
    V, vx, fell, x = best
    return dict(setting=name, best_V=V, best_vx=vx, best_fell=fell, seeds=seeds,
                n_trials=n_trials,
                params=" ".join(f"{v:.4f}" for v in x.ravel()))


def stage_bo(names, n_trials, seeds, workers, duration):
    jobs = [(n, n_trials, seeds, duration) for n in names]
    print(f"BO recoverability: {names}, {n_trials} trials x {seeds} seeds each",
          flush=True)
    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_bo_job, jobs):
            rows.append({k: str(v) for k, v in row.items()})
            print(f"== {row['setting']}: best V={row['best_V']:.3f} "
                  f"vx={row['best_vx']:.2f} fell={row['best_fell']}/{row['seeds']}",
                  flush=True)
    if os.path.exists(BO_CSV):
        with open(BO_CSV) as f:
            olds = [r for r in csv.DictReader(f)
                    if r["setting"] not in {r2["setting"] for r2 in rows}]
        rows.extend(olds)
    with open(BO_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {BO_CSV}")


# ── Report ────────────────────────────────────────────────────────────────────

def _sweep_summary():
    if not os.path.exists(SWEEP_CSV):
        return {}
    with open(SWEEP_CSV) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for name in BY_NAME:
        rs = [r for r in rows if r["setting"] == name]
        if not rs:
            continue
        fell = [int(r["fell"]) for r in rs]
        shocks = [int(r["shock"]) for r in rs]
        fa = [float(r["fall_after_shift"]) for r in rs
              if r["fall_after_shift"] not in ("", "nan")]
        vx2 = [float(r["mean_vx_ph2"]) for r in rs
               if r["mean_vx_ph2"] not in ("", "nan")]
        out[name] = dict(
            n=len(rs), fall_rate=float(np.mean(fell)),
            shock_rate=(float(np.mean(shocks))),
            med_fall_after=(float(np.median(fa)) if fa else np.nan),
            mean_vx_ph2=(float(np.mean(vx2)) if vx2 else np.nan))
    return out


def report():
    summ = _sweep_summary()
    if summ:
        print("\n== no-adapt fall-rate sweep ==")
        print(f"{'setting':<18} {'n':>3} {'fall%':>6} {'shock%':>7} "
              f"{'med fall t+':>11} {'vx ph2':>7}")
        for name, s in summ.items():
            print(f"{name:<18} {s['n']:>3} {100*s['fall_rate']:>5.0f}% "
                  f"{100*s['shock_rate']:>6.0f}% {s['med_fall_after']:>10.1f}s "
                  f"{s['mean_vx_ph2']:>7.2f}")
    if os.path.exists(SCREEN_CSV):
        with open(SCREEN_CSV) as f:
            rows = list(csv.DictReader(f))
        print("\n== recoverability screen (always-shifted, gait fixed) ==")
        for name in sorted({r["setting"] for r in rows}):
            rs = [r for r in rows if r["setting"] == name]
            gaits = sorted({r["gait"] for r in rs})
            surv = []
            for g in gaits:
                gr = [r for r in rs if r["gait"] == g]
                ok = all(int(r["fell"]) == 0 for r in gr)
                vx = np.mean([float(r["mean_vx_tail"]) for r in gr])
                if ok and vx >= FWD_VX:
                    surv.append((g, vx))
            best = max(surv, key=lambda t: t[1]) if surv else None
            print(f"{name:<18} forward survivors {len(surv)}/{len(gaits)}"
                  + (f", best vx={best[1]:.2f} ({best[0]})" if best
                     else "  -> looks UNRECOVERABLE"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sweep", "screen", "bo", "report"])
    ap.add_argument("--bo-trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--duration", type=float, default=SWEEP_DURATION)
    ap.add_argument("--gaits", type=int, default=12,
                    help="screen: number of LHS gaits (plus the incumbent)")
    ap.add_argument("--min-fall-rate", type=float, default=0.5,
                    help="screen: only settings whose sweep fall rate >= this")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these setting names")
    args = ap.parse_args()
    if args.stage == "sweep":
        stage_sweep(args.seeds, args.workers, args.duration, only=args.only)
    elif args.stage == "screen":
        stage_screen(args.gaits, min(args.seeds, 3), args.workers,
                     args.min_fall_rate, only=args.only)
    elif args.stage == "bo":
        stage_bo(args.only or ["B_8kg_0.225"], args.bo_trials,
                 min(args.seeds, 3), args.workers, SCREEN_DURATION)
    else:
        report()


if __name__ == "__main__":
    main()
