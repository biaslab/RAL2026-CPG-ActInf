"""Fit a per-phase CPG optimum for the payload experiment and run the
CROSS-PENALTY SCREEN.

Two conditions, each held fixed over the scored tail of the bout (no trigger):
  * centered -> payload centered over the trunk (phase 1 of the main experiment);
  * shifted  -> payload ramps to the offset position at t=1 s, WHILE WALKING
    (phase 2). It is not attached pre-shifted: settling a crouched robot under
    the full offset is statically unstable, whereas the real phase 2 is always
    entered walking, with the attitude feedback active.

Each condition's optimum is fit by GP-UCB Bayesian optimisation (incumbent
injected as probe 0, so the optimum never regresses below it), scoring each
candidate by the paper's stability criterion V over the steady-state part of
the bout (t >= SKIP_T), averaged over seeds, with the SAME stepping as the
main experiment (attitude feedback ON, payload attached during settling).

Besides the two optima the script evaluates the full 3x2 CROSS matrix
(incumbent / centered-optimum / shifted-optimum, each under both conditions)
and prints the off-diagonal penalty. This is the cheap go/no-go screen: if
V(centered-opt | shifted) is NOT clearly worse than V(shifted-opt | shifted),
there is no room for adaptation and the main experiment is not worth running.

Usage (from repo root):
    python experiment-simulation/experiment-payload-adapt/fit_payload_oracles.py [--trials 60 --seeds 3 --workers 2]
Output: experiment-simulation/experiment-payload-adapt/results/payload_optima.json
"""

import argparse
import importlib.util
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# experiment-simulation/experiment-payload-adapt/ -> repo root (two levels up)
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS = os.path.join(_HERE, "results")
FIT_DURATION = 20.0           # single-condition bout length [s]
SHIFT_EARLY_T = 1.0           # `shifted` condition: shift this soon, while walking
SKIP_T = 5.0                  # score only the steady-state tail (t >= SKIP_T)
TARGET_VX = 0.5
ATT_REF_DEG = 10.0
V_FALL = -2.0
CONDITIONS = ["centered", "shifted"]


def _pl():
    spec = importlib.util.spec_from_file_location(
        "pl_fit", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cond_cfg(pl, cond):
    pc = pl.payload_defaults()
    # centered: never shifts; shifted: shifts at t=1 s, while walking.
    pc["t_shift"] = None if cond == "centered" else SHIFT_EARLY_T
    return pc


def _lowpass(x, w=50):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x
    w = min(w, len(x))
    c = np.concatenate([[0.0], np.cumsum(x)])
    return np.array([(c[i + 1] - c[max(0, i - w + 1)]) / (i + 1 - max(0, i - w + 1))
                     for i in range(len(x))])


def _score_tail(res):
    """Stability criterion V over the steady-state tail (t >= SKIP_T)."""
    if int(res["fell"]):
        return V_FALL
    t = np.asarray(res["t"])
    tail = np.nonzero(t >= SKIP_T)[0]
    if len(tail) < 200:
        return V_FALL
    k0 = int(tail[0])
    vx = np.asarray(res["vx"])[k0:]
    roll = np.asarray(res["roll"])[k0:]
    pitch = np.asarray(res["pitch"])[k0:]
    r_v = min(max(float(np.mean(vx)), 0.0) / TARGET_VX, 1.0)
    rms_roll = np.rad2deg(np.sqrt(np.mean(_lowpass(roll) ** 2)))
    lp = _lowpass(pitch)
    rms_pitch = np.rad2deg(np.sqrt(np.mean((lp - np.median(lp)) ** 2)))
    return r_v - (rms_roll + rms_pitch) / ATT_REF_DEG


def _score_params(pl, box, cond, params, seeds):
    """Mean V for a fixed gait under `cond`, over `seeds` seeds."""
    vs = []
    for s in range(seeds):
        res = pl.run_trial(s, "noadapt", 1.0, np.asarray(params, float), box,
                           _cond_cfg(pl, cond), trigger="ce",
                           duration=FIT_DURATION)
        vs.append(_score_tail(res))
    return float(np.mean(vs))


def _bo_condition(job):
    """Bayesian-optimize the gait for one payload condition (GP-UCB, feedback ON)."""
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = True
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    from methods.bo_optimizer import BOOptimizer, BetaSchedule
    from methods.cpg_bounds import bounds_lower, bounds_upper
    cond, incumbent, n_trials, seeds = job
    pl = _pl()
    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    box = (lo, hi)
    bo = BOOptimizer(
        bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
        beta_schedule=BetaSchedule(beta_init=5.0, beta_min=1.0,
                                   n_decay_start=max(10, n_trials // 2), gamma=0.9),
        n_init=8, seed=hash(cond) % 9973)
    rng = np.random.default_rng(hash(cond) % 9973)
    inc_V = _score_params(pl, box, cond, incumbent, seeds)
    bo._append(np.asarray(incumbent, float), inc_V)      # incumbent as probe 0
    best_V, best_x = inc_V, np.asarray(incumbent, float)
    for t in range(n_trials):
        if t < 8:
            x = rng.uniform(lo, hi)
        else:
            try:
                model = bo.fit_model()
                x = bo.from_unit(bo.suggest(model, bo.beta_schedule(t)))
            except Exception:
                x = rng.uniform(lo, hi)
        x = np.clip(np.asarray(x, float), lo, hi)
        V = _score_params(pl, box, cond, x, seeds)
        bo._append(x, V)
        if V > best_V:
            best_V, best_x = V, x
    return cond, best_x.tolist(), best_V, inc_V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cross-seeds", type=int, default=6,
                    help="seeds for the final cross-penalty evaluation")
    a = ap.parse_args()

    # Flat-optimal incumbent: results/incumbent.json, falling back to the
    # hard-coded copy in run_experiment.py (keeps the folder self-contained; the
    # original experiment-flat BO fit now lives under archive/experiments/).
    incumbent = _pl().load_incumbent()

    os.makedirs(RESULTS, exist_ok=True)
    jobs = [(cond, incumbent, a.trials, a.seeds) for cond in CONDITIONS]
    ctx = get_context("spawn")
    optima = {}
    print(f"BO per payload condition: {a.trials} trials x {a.seeds} seeds, feedback ON")
    with ctx.Pool(min(a.workers, len(jobs)), maxtasksperchild=1) as pool:
        for cond, best_x, best_V, inc_V in pool.imap_unordered(_bo_condition, jobs):
            beat = best_V > inc_V + 1e-6
            optima[cond] = {"params": best_x, "mean_V": best_V,
                            "incumbent_V": inc_V, "beats_incumbent": bool(beat)}
            print(f"[{cond:8s}] BO best mean_V={best_V:+.3f} | incumbent={inc_V:+.3f} | "
                  f"{'BETTER gait found (+%.3f)' % (best_V - inc_V) if beat else 'incumbent not beaten'}",
                  flush=True)

    # ── Cross-penalty screen: every gait under every condition ───────────────
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = True
    from methods.cpg_bounds import bounds_lower, bounds_upper
    box = (bounds_lower.numpy(), bounds_upper.numpy())
    pl = _pl()
    gaits = {"incumbent": incumbent,
             "centered_opt": np.asarray(optima["centered"]["params"], float),
             "shifted_opt": np.asarray(optima["shifted"]["params"], float)}
    print(f"\ncross-penalty matrix (mean V over {a.cross_seeds} seeds):")
    cross = {}
    for gname, gx in gaits.items():
        cross[gname] = {}
        for cond in CONDITIONS:
            cross[gname][cond] = _score_params(pl, box, cond, gx, a.cross_seeds)
        print(f"  {gname:12s} | " + " | ".join(
            f"{c}: {cross[gname][c]:+.3f}" for c in CONDITIONS), flush=True)

    gap = cross["shifted_opt"]["shifted"] - cross["centered_opt"]["shifted"]
    print(f"\nSCREEN: room for adaptation on `shifted` = "
          f"V(shifted_opt|shifted) - V(centered_opt|shifted) = {gap:+.3f}")
    print("  -> " + ("GO: sustained mismatch exists, the main experiment can pay off."
                     if gap > 0.15 else
                     "NO-GO: the phase-1 gait is already near-optimal under the "
                     "shifted payload; increase --mass / shift or pick another challenge."))

    out = os.path.join(RESULTS, "payload_optima.json")
    with open(out, "w") as f:
        json.dump({**optima, "cross": cross, "screen_gap": gap}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
