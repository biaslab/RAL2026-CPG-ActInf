"""Fit a per-phase CPG optimum for the leg-damage experiment and run the
CROSS-PENALTY SCREEN.

Two conditions, each held fixed over the scored tail of the bout (no trigger):
  * healthy -> all legs at full torque (phase 1 of the main experiment);
  * damaged -> one leg's hip+knee maxForce ramps to `damage_force` at t=1 s,
    WHILE WALKING (phase 2). It is not damaged from rest: the real phase 2 is
    always entered walking, with the attitude feedback active.

Each condition's optimum is fit by GP-UCB Bayesian optimisation (incumbent
injected as probe 0, so the optimum never regresses below it), scoring each
candidate by the paper's stability criterion V over the steady-state part of the
bout (t >= SKIP_T), averaged over seeds, with the SAME stepping as the main
experiment (attitude feedback ON).

Besides the two optima the script evaluates the full 3x2 CROSS matrix (incumbent
/ healthy-optimum / damaged-optimum, each under both conditions) and prints the
off-diagonal penalty. This is the cheap go/no-go screen AND the direct answer to
"is there a gap between no-adapt (incumbent) and oracle (damaged-optimum) under
the damaged leg?": if V(incumbent | damaged) is NOT clearly worse than
V(damaged_opt | damaged), the GLOBAL/symmetric parameterisation cannot express a
compensating gait and the main experiment is not worth running (see notes/
adaptation-challenge-candidates.md #3 -- the parameterisation, not the damage,
would then be the bottleneck).

Usage (from repo root):
    python experiment-damage-adapt/fit_damage_oracles.py [--trials 60 --seeds 3 --workers 2]
Output: experiment-damage-adapt/results/damage_optima.json
"""

import argparse
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

RESULTS = os.path.join(_HERE, "results")
FIT_DURATION = 20.0           # single-condition bout length [s]
DAMAGE_EARLY_T = 1.0          # `damaged` condition: damage this soon, while walking
SKIP_T = 5.0                  # score only the steady-state tail (t >= SKIP_T)
TARGET_VX = 0.5
ATT_REF_DEG = 10.0
V_FALL = -2.0
CONDITIONS = ["healthy", "damaged"]


def _dm():
    spec = importlib.util.spec_from_file_location(
        "dm_fit", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cond_cfg(dm, cond):
    dc = dm.damage_defaults()
    # healthy: never damaged; damaged: damage ramps at t=1 s, while walking.
    dc["t_damage"] = None if cond == "healthy" else DAMAGE_EARLY_T
    return dc


def _lowpass(x, w=50):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x
    w = min(w, len(x))
    c = np.concatenate([[0.0], np.cumsum(x)])
    return np.array([(c[i + 1] - c[max(0, i - w + 1)]) / (i + 1 - max(0, i - w + 1))
                     for i in range(len(x))])


def _score_tail(res, shaped=False):
    """Stability criterion V over the steady-state tail (t >= SKIP_T).

    Under the damaged leg, most of the search box FALLS, which clamps V to a
    flat V_FALL plateau with no gradient -- GP-UCB then never escapes the
    incumbent probe (this is exactly what happened in the first, unshaped fit).
    With `shaped=True` a fallen run instead earns partial credit for how long it
    stayed up (survival fraction), mapped to [V_FALL, V_FALL+1] = [-2, -1], so a
    gait that survives longer scores higher and BO can climb out of the plateau
    toward the (rare) fully-surviving basin. `shaped` is a guidance surrogate
    ONLY; the reported optima / cross matrix always use the true (unshaped) V, so
    a fallen gait can never masquerade as a survivor (survivors are >= -1)."""
    if int(res["fell"]):
        if not shaped:
            return V_FALL
        t_end = float(res["t"][-1]) if len(res["t"]) else 0.0
        return V_FALL + min(1.0, t_end / FIT_DURATION)   # in [-2, -1]
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


def _score_params(dm, box, cond, params, seeds, shaped=False):
    """Mean V for a fixed gait under `cond`, over `seeds` seeds."""
    vs = []
    for s in range(seeds):
        res = dm.run_trial(s, "noadapt", 1.0, np.asarray(params, float), box,
                           _cond_cfg(dm, cond), trigger="ce",
                           duration=FIT_DURATION)
        vs.append(_score_tail(res, shaped=shaped))
    return float(np.mean(vs))


def _bo_condition(job):
    """Bayesian-optimize the gait for one damage condition (GP-UCB, feedback ON).

    BO is guided by the SHAPED score (survival-time credit for falls) so it can
    climb out of the flat fall plateau; `best_x` is the argmax of the shaped
    score and its TRUE (unshaped) mean_V is returned for reporting. `seed_probes`
    are extra initial designs (besides the incumbent) injected before the GP
    starts -- used to seed the damaged fit with the healthy optimum, which is a
    known in-box survivor of the weak leg."""
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = True
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    from methods.bo_optimizer import BOOptimizer, BetaSchedule
    from methods.cpg_bounds import bounds_lower, bounds_upper
    cond, incumbent, n_trials, seeds, seed_probes = job
    dm = _dm()
    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    box = (lo, hi)
    bo = BOOptimizer(
        bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
        beta_schedule=BetaSchedule(beta_init=5.0, beta_min=1.0,
                                   n_decay_start=max(10, n_trials // 2), gamma=0.9),
        n_init=8, seed=hash(cond) % 9973)
    rng = np.random.default_rng(hash(cond) % 9973)

    def _both(x):                                        # (shaped_for_BO, true_V)
        return (_score_params(dm, box, cond, x, seeds, shaped=True),
                _score_params(dm, box, cond, x, seeds, shaped=False))

    inc_sh, inc_V = _both(incumbent)
    bo._append(np.asarray(incumbent, float), inc_sh)     # incumbent as probe 0
    best_sh, best_x, best_V = inc_sh, np.asarray(incumbent, float), inc_V
    for pr in (seed_probes or []):                       # extra survivor seeds
        pr = np.clip(np.asarray(pr, float), lo, hi)
        sh, V = _both(pr)
        bo._append(pr, sh)
        if sh > best_sh:
            best_sh, best_x, best_V = sh, pr, V
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
        sh, V = _both(x)
        bo._append(x, sh)
        if sh > best_sh:
            best_sh, best_x, best_V = sh, x, V
    return cond, best_x.tolist(), best_V, inc_V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cross-seeds", type=int, default=6,
                    help="seeds for the final cross-penalty evaluation")
    a = ap.parse_args()

    incumbent = np.asarray(json.load(open(os.path.join(
        _REPO, "experiment-flat", "results", "selected_params.json")))["params"], float)

    os.makedirs(RESULTS, exist_ok=True)
    ctx = get_context("spawn")
    optima = {}
    dm0 = _dm()
    dc0 = dm0.damage_defaults()
    print(f"leg-damage oracle fit: leg={dc0['leg']} hip+knee "
          f"{dc0['healthy_force']}->{dc0['damage_force']} Nm")
    print(f"BO per condition: {a.trials} trials x {a.seeds} seeds, feedback ON")

    # Fit HEALTHY first, then DAMAGED seeded with the healthy optimum. Under the
    # weak leg most of the box falls (flat plateau), so BO needs a known in-box
    # survivor to climb from; the healthy optimum -- a slightly slower-swing gait
    # -- is one, and seeding it lets the damaged fit escape the incumbent.
    def _run(cond, seed_probes=()):
        with ctx.Pool(1, maxtasksperchild=1) as pool:
            return pool.apply(_bo_condition,
                              ((cond, incumbent, a.trials, a.seeds, seed_probes),))

    cond, best_x, best_V, inc_V = _run("healthy")
    optima["healthy"] = {"params": best_x, "mean_V": best_V,
                         "incumbent_V": inc_V, "beats_incumbent": bool(best_V > inc_V + 1e-6)}
    print(f"[healthy ] BO best mean_V={best_V:+.3f} | incumbent={inc_V:+.3f} | "
          f"{'BETTER gait found (+%.3f)' % (best_V - inc_V) if best_V > inc_V + 1e-6 else 'incumbent not beaten'}",
          flush=True)

    cond, best_x, best_V, inc_V = _run("damaged",
                                       seed_probes=[optima["healthy"]["params"]])
    optima["damaged"] = {"params": best_x, "mean_V": best_V,
                         "incumbent_V": inc_V, "beats_incumbent": bool(best_V > inc_V + 1e-6)}
    print(f"[damaged ] BO best mean_V={best_V:+.3f} | incumbent={inc_V:+.3f} | "
          f"{'BETTER gait found (+%.3f)' % (best_V - inc_V) if best_V > inc_V + 1e-6 else 'incumbent not beaten'}",
          flush=True)

    # ── Cross-penalty screen: every gait under every condition ───────────────
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = True
    from methods.cpg_bounds import bounds_lower, bounds_upper
    box = (bounds_lower.numpy(), bounds_upper.numpy())
    dm = _dm()
    gaits = {"incumbent": incumbent,
             "healthy_opt": np.asarray(optima["healthy"]["params"], float),
             "damaged_opt": np.asarray(optima["damaged"]["params"], float)}
    print(f"\ncross-penalty matrix (mean V over {a.cross_seeds} seeds):")
    cross = {}
    for gname, gx in gaits.items():
        cross[gname] = {}
        for cond in CONDITIONS:
            cross[gname][cond] = _score_params(dm, box, cond, gx, a.cross_seeds)
        print(f"  {gname:12s} | " + " | ".join(
            f"{c}: {cross[gname][c]:+.3f}" for c in CONDITIONS), flush=True)

    # The damaged oracle is the best re-tuned gait UNDER the damaged leg,
    # evaluated by true V (either the dedicated damaged fit or the healthy
    # optimum, whichever survives the weak leg better). Promote it so the
    # oracle arm of run_experiment.py loads the genuine survivor.
    best_retuned = max(("healthy_opt", "damaged_opt"),
                       key=lambda g: cross[g]["damaged"])
    if cross[best_retuned]["damaged"] > cross["damaged_opt"]["damaged"] + 1e-9:
        optima["damaged"] = {"params": gaits[best_retuned].tolist(),
                             "mean_V": cross[best_retuned]["damaged"],
                             "incumbent_V": optima["damaged"]["incumbent_V"],
                             "beats_incumbent": True,
                             "source": best_retuned}

    gap = cross[best_retuned]["damaged"] - cross["incumbent"]["damaged"]
    print(f"\nSCREEN: room for adaptation on `damaged` = "
          f"V(best re-tuned gait|damaged) - V(incumbent|damaged) = {gap:+.3f} "
          f"({cross[best_retuned]['damaged']:+.3f} via {best_retuned} "
          f"vs {cross['incumbent']['damaged']:+.3f})")
    print("  -> " + ("GO: the global CPG params CAN express a compensating gait; "
                     "the main experiment can pay off."
                     if gap > 0.15 else
                     "NO-GO: the incumbent is already ~as good as any global gait "
                     "under the weak leg -- the parameterisation, not the damage, "
                     "is the bottleneck (increase --damage severity or pick another "
                     "challenge)."))

    out = os.path.join(RESULTS, "damage_optima.json")
    with open(out, "w") as f:
        json.dump({**optima, "cross": cross, "screen_gap": gap}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
