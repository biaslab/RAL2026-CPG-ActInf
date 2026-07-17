"""Fit the per-leg leg-damage recovery gait (the oracle arm's target).

Searches the 4 per-leg hip amplitudes (methods.marxefe_optimizer.PerLegCPG) for a
gait that survives SUSTAINED damage of the target leg and travels as far as
possible, and writes it to results/damage_optima.json under "damaged" (11-D). The
"healthy" entry is the symmetric incumbent (also 11-D). The continual experiment's
oracle arm loads "damaged" from this file.

This supersedes the old global-8-D BO oracle fit: single-leg damage is only
recoverable with per-leg control (the asymmetry is irreducible for a global gait).

Usage (from repo root):
    python experiment-damage-adapt/fit_damage_oracles.py            # 20 Nm, RR
    python experiment-damage-adapt/fit_damage_oracles.py --force 20 --n 60 --seeds 4
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FIT_HOLD = 18.0          # sustained-damage bout length for scoring [s]


def _dm():
    spec = importlib.util.spec_from_file_location(
        "dm_run", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _trial(dm, params, force, seed, hold=FIT_HOLD):
    """Sustained damage from t=3 s; return (fwd distance, RMS tilt deg) if it
    survives the hold, else None."""
    ph = dm.DamagePhysics(damage_force=force)
    cpg = ph.setup(seed=seed)
    DT = 0.01
    roll = pitch = 0.0
    y0 = None
    yend = 0.0
    rr, pp = [], []
    for k in range(int((3.0 + hold) / DT)):
        t = k * DT
        frac = 0.0 if t < 3.0 else min(1.0, (t - 3.0) / 1.0)
        st = ph.actuate(cpg, params, roll, pitch, frac)
        roll, pitch = st.roll, st.pitch
        if t >= 4.5:
            if y0 is None:
                y0 = st.base_pos[1]
            yend = st.base_pos[1]
            rr.append(roll); pp.append(pitch)
        if st.fell:
            ph.disconnect()
            return None
    ph.disconnect()
    tilt = float(np.rad2deg(np.sqrt(np.mean(np.array(rr) ** 2 + np.array(pp) ** 2)))) \
        if rr else 99.0
    return (yend - (y0 or 0.0), tilt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", type=float, default=None,
                    help="damage force [Nm] (default: the experiment's DAMAGE_FORCE)")
    ap.add_argument("--n", type=int, default=48, help="Latin-hypercube candidates")
    ap.add_argument("--seeds", type=int, default=4,
                    help="seeds each candidate must ALL survive (robustness)")
    a = ap.parse_args()

    from scipy.stats.qmc import LatinHypercube
    from methods.marxefe_optimizer import PerLegCPG
    from methods.cpg_bounds import bounds_lower, bounds_upper
    dm = _dm()
    PerLegCPG.ATTITUDE_FEEDBACK = True
    force = a.force if a.force is not None else dm.DAMAGE_FORCE
    lo, hi = PerLegCPG.expand_box(bounds_lower.numpy(), bounds_upper.numpy())
    inc = dm.load_incumbent()
    free = dm.FREE_DIMS_DAMAGE                       # the 4 per-leg hip amplitudes

    print(f"fitting per-leg recovery gait: {dm.DAMAGE_LEG} @ {force:g} Nm, "
          f"searching {[dm.PARAM_NAMES[j] for j in free]}, "
          f"{a.n} candidates x {a.seeds} seeds (all must survive)")

    lhs = lo[free] + LatinHypercube(d=len(free), seed=3).random(n=a.n) * (hi[free] - lo[free])
    best = None
    for i, cred in enumerate(lhs):
        x = inc.copy(); x[free] = cred
        res = [_trial(dm, x, force, s) for s in range(a.seeds)]
        if any(r is None for r in res):
            continue
        dist = float(np.mean([r[0] for r in res]))
        tilt = float(np.mean([r[1] for r in res]))
        score = dist - tilt / 10.0
        if best is None or score > best[0]:
            best = (score, x.copy(), dist, tilt)
            print(f"[{i:3d}] survive {a.seeds}/{a.seeds}  dist {dist:5.2f} m  "
                  f"tilt {tilt:4.1f} deg  score {score:+.2f}  <- best", flush=True)

    if best is None:
        raise SystemExit(f"no gait survived all {a.seeds} seeds at {force:g} Nm; "
                         "lower the force or widen the search")

    inc_falls = sum(1 for s in range(a.seeds) if _trial(dm, inc, force, s) is None)
    print(f"\nincumbent (symmetric) falls {inc_falls}/{a.seeds} at {force:g} Nm")
    print("recovery gait per-leg hipA:",
          {dm.PARAM_NAMES[j]: round(float(best[1][j]), 3) for j in free})

    out = dm.OPTIMA_JSON
    d = json.load(open(out)) if os.path.exists(out) else {}
    d["damaged"] = {"params": best[1].tolist(), "mean_dist": best[2],
                    "mean_tilt_deg": best[3], "force": float(force),
                    "note": "per-leg hip-amplitude recovery gait (11-D PerLegCPG)"}
    d["healthy"] = {"params": inc.tolist(), "note": "symmetric incumbent (11-D)"}
    with open(out, "w") as f:
        json.dump(d, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
