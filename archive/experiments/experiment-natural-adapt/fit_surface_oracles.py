"""Fit a per-surface CPG optimum for each natural surface, with the VMC attitude
feedback ON, on that surface's REAL geometry and friction (not a flat plane).

Each surface (grass/gravel/rocks/river/ice) is fit on a single-surface transect
(a short grass lead-in, then a long band of that surface) using the SAME stepping
as the main experiment (per-step friction, attitude feedback, terrain-relative
fall check): we hold a candidate 8-vector for the whole bout and score it by the
paper's stability criterion V over the portion walked ON the surface. A
Latin-hypercube candidate set is scored over several seeds; the best mean-V
candidate is the surface optimum. These optima define the clairvoyant per-band
`oracle` arm (see run_experiment.py) that upper-bounds what any parameter switch
could achieve.

Usage (from repo root):
    python experiment-natural-adapt/fit_surface_oracles.py [--cands 64 --seeds 4 --workers 11]
Output: experiment-natural-adapt/results/surface_optima.json
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
LEAD_IN = 4.0                 # grass lead-in before the surface band [m]
REACH = 30.0                  # single-surface transect length [m]
TARGET_VX = 0.5
ATT_REF_DEG = 10.0
V_FALL = -2.0
SURFACES = ["grass", "gravel", "rocks", "river", "ice"]


def _nat():
    spec = importlib.util.spec_from_file_location(
        "nat_fit", os.path.join(_HERE, "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _single_cfg(surf, seed):
    from methods import terrain
    mu = terrain.NATURAL_SURFACES[surf][0]
    gmu = terrain.NATURAL_SURFACES["grass"][0]
    bands = [(0.0, "grass"), (LEAD_IN, surf)]
    zones = [(0.0, gmu, "grass"), (LEAD_IN, mu, surf)]
    return {"kind": "natural", "bands": bands, "zones": zones,
            "band_slopes": [0.0, 0.0], "band_elev0": [0.0, 0.0],
            "base_mu": gmu, "reach": REACH, "seed": int(seed)}


def _lowpass(x, w=50):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x
    w = min(w, len(x))
    c = np.concatenate([[0.0], np.cumsum(x)])
    return np.array([(c[i + 1] - c[max(0, i - w + 1)]) / (i + 1 - max(0, i - w + 1))
                     for i in range(len(x))])


def _score_on_surface(res):
    """Stability criterion V over the portion walked on the surface band (y>=LEAD_IN)."""
    if int(res["fell"]):
        return V_FALL
    y = np.asarray(res["y"])
    onto = np.nonzero(y >= LEAD_IN)[0]
    if len(onto) == 0:
        return V_FALL
    k0 = int(onto[0])
    vx = np.asarray(res["vx"])[k0:]
    roll = np.asarray(res["roll"])[k0:]
    pitch = np.asarray(res["pitch"])[k0:]
    if len(vx) < 200:
        return V_FALL
    r_v = min(max(float(np.mean(vx)), 0.0) / TARGET_VX, 1.0)
    rms_roll = np.rad2deg(np.sqrt(np.mean(_lowpass(roll) ** 2)))
    lp = _lowpass(pitch)
    rms_pitch = np.rad2deg(np.sqrt(np.mean((lp - np.median(lp)) ** 2)))
    return r_v - (rms_roll + rms_pitch) / ATT_REF_DEG


def _score_params(nat, box, surf, params, seeds):
    """Mean stability criterion V for a fixed gait on `surf`, over `seeds` seeds."""
    vs = []
    for s in range(seeds):
        res = nat.run_trial(s, "noadapt", 1.0, np.asarray(params, float),
                            _single_cfg(surf, s), box, trigger="cusum")
        vs.append(_score_on_surface(res))
    return float(np.mean(vs))


def _bo_surface(job):
    """Bayesian-optimize the CPG gait for one surface (GP-UCB, feedback ON), each
    candidate scored on the surface's real geometry+friction. The incumbent is
    injected as the first probe so the returned optimum never regresses below it.
    Returns (surface, best_params, best_meanV, incumbent_meanV)."""
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = True
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    from methods.bo_optimizer import BOOptimizer, BetaSchedule
    from methods.cpg_bounds import bounds_lower, bounds_upper
    surf, incumbent, n_trials, seeds = job
    nat = _nat()
    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    box = (lo, hi)
    bo = BOOptimizer(
        bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
        beta_schedule=BetaSchedule(beta_init=5.0, beta_min=1.0,
                                   n_decay_start=max(10, n_trials // 2), gamma=0.9),
        n_init=8, seed=hash(surf) % 9973)
    rng = np.random.default_rng(hash(surf) % 9973)
    inc_V = _score_params(nat, box, surf, incumbent, seeds)
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
        V = _score_params(nat, box, surf, x, seeds)
        bo._append(x, V)
        if V > best_V:
            best_V, best_x = V, x
    return surf, best_x.tolist(), best_V, inc_V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    incumbent = np.asarray(json.load(open(os.path.join(
        _REPO, "experiment-flat", "results", "selected_params.json")))["params"], float)

    os.makedirs(RESULTS, exist_ok=True)
    jobs = [(surf, incumbent, a.trials, a.seeds) for surf in SURFACES]
    ctx = get_context("spawn")
    optima = {}
    print(f"BO per surface: {a.trials} trials x {a.seeds} seeds, feedback ON")
    with ctx.Pool(min(a.workers, len(jobs)), maxtasksperchild=1) as pool:
        for surf, best_x, best_V, inc_V in pool.imap_unordered(_bo_surface, jobs):
            beat = best_V > inc_V + 1e-6
            optima[surf] = {"params": best_x, "mean_V": best_V,
                            "incumbent_V": inc_V, "beats_incumbent": bool(beat)}
            print(f"[{surf:7s}] BO best mean_V={best_V:+.3f} | incumbent={inc_V:+.3f} | "
                  f"{'BETTER gait found (+%.3f)' % (best_V - inc_V) if beat else 'incumbent not beaten'}",
                  flush=True)

    out = os.path.join(RESULTS, "surface_optima.json")
    with open(out, "w") as f:
        json.dump(optima, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
