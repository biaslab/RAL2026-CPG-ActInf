"""Ceiling test for adapting the ATTITUDE-FEEDBACK GAINS across terrains.

The whole project so far adapts the CPG's 8 gait-SHAPE parameters, and even a
clairvoyant per-band oracle over those cannot beat holding the incumbent (the
optimum shift does not transfer through the transitions, and switching the gait
shape is itself destabilizing). This module asks the prerequisite question for
the pivot: is there ANY headroom in adapting the 4 attitude-feedback gains
[kp_roll, kd_roll, kp_pitch, kd_pitch] instead? Those modulate the balance loop
continuously (no gait-phase discontinuity), so switching them is gentle.

Two stages:
  * fit   -> per surface (grass/gravel/rocks/river/ice), Bayesian-optimize the 4
             gains (CPG shape fixed at the incumbent) on that surface's real
             geometry+friction; write results/surface_gain_optima.json;
  * compare-> on the natural transect, no-adapt (fixed default gains everywhere)
             vs oracle (clairvoyant per-band gains). If the gain oracle beats
             no-adapt, adapting the gains has headroom and the pivot is worth
             pursuing; if it ties, the balance loop is already terrain-robust.

Usage (from repo root):
    python experiment-gain-adapt/ceiling.py fit      [--trials 40 --seeds 3]
    python experiment-gain-adapt/ceiling.py compare  [--seeds 20]
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
GAIN_JSON = os.path.join(RESULTS, "surface_gain_optima.json")
LEAD_IN = 4.0
FIT_REACH = 30.0
COURSE_REACH = 45.0
COURSE_DURATION = 120.0
SURFACES = ["grass", "gravel", "rocks", "river", "ice"]
GAIN_RAMP = 100          # steps (1 s) to morph the gains toward a new band's optimum
TARGET_VX = 0.5
ATT_REF_DEG = 10.0
V_FALL = -2.0


def _nat():
    spec = importlib.util.spec_from_file_location(
        "nat_gain", os.path.join(_REPO, "experiment-natural-adapt", "run_experiment.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _single_cfg(surf, seed):
    from methods import terrain
    mu = terrain.NATURAL_SURFACES[surf][0]
    gmu = terrain.NATURAL_SURFACES["grass"][0]
    return {"kind": "natural", "bands": [(0.0, "grass"), (LEAD_IN, surf)],
            "zones": [(0.0, gmu, "grass"), (LEAD_IN, mu, surf)],
            "band_slopes": [0.0, 0.0], "band_elev0": [0.0, 0.0],
            "base_mu": gmu, "reach": FIT_REACH, "seed": int(seed)}


def _lowpass(x, w=50):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x
    w = min(w, len(x))
    c = np.concatenate([[0.0], np.cumsum(x)])
    return np.array([(c[i + 1] - c[max(0, i - w + 1)]) / (i + 1 - max(0, i - w + 1))
                     for i in range(len(x))])


def _score_on_surface(res):
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


def _score_gains(nat, surf, gains, incumbent, box, seeds):
    vs = []
    for s in range(seeds):
        res = nat.run_trial(s, "noadapt", 1.0, incumbent, _single_cfg(surf, s),
                            box, trigger="cusum", gain_policy=np.asarray(gains, float))
        vs.append(_score_on_surface(res))
    return float(np.mean(vs))


def _bo_gains(job):
    from methods.marxefe_optimizer import JointCPG, GAIN_DEFAULT, GAIN_LOWER, GAIN_UPPER
    JointCPG.ATTITUDE_FEEDBACK = True
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    import torch
    from methods.bo_optimizer import BOOptimizer, BetaSchedule
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    surf, incumbent, n_trials, seeds = job
    nat = _nat()
    box = (bl.numpy(), bu.numpy())
    lo, hi = GAIN_LOWER, GAIN_UPPER
    bo = BOOptimizer(
        bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
        beta_schedule=BetaSchedule(beta_init=4.0, beta_min=1.0,
                                   n_decay_start=max(8, n_trials // 2), gamma=0.9),
        n_init=6, seed=hash(surf) % 9973)
    rng = np.random.default_rng(hash(surf) % 9973)
    inc = np.asarray(incumbent, float)
    def_V = _score_gains(nat, surf, GAIN_DEFAULT, inc, box, seeds)
    bo._append(np.asarray(GAIN_DEFAULT, float), def_V)         # default gains as probe 0
    best_V, best_g = def_V, np.asarray(GAIN_DEFAULT, float)
    for t in range(n_trials):
        if t < 6:
            g = rng.uniform(lo, hi)
        else:
            try:
                model = bo.fit_model()
                g = bo.from_unit(bo.suggest(model, bo.beta_schedule(t)))
            except Exception:
                g = rng.uniform(lo, hi)
        g = np.clip(np.asarray(g, float), lo, hi)
        V = _score_gains(nat, surf, g, inc, box, seeds)
        bo._append(g, V)
        if V > best_V:
            best_V, best_g = V, g
    return surf, best_g.tolist(), best_V, def_V


def do_fit(trials, seeds, workers):
    incumbent = np.asarray(json.load(open(os.path.join(
        _REPO, "experiment-flat", "results", "selected_params.json")))["params"], float)
    os.makedirs(RESULTS, exist_ok=True)
    jobs = [(surf, incumbent, trials, seeds) for surf in SURFACES]
    ctx = get_context("spawn")
    optima = {}
    print(f"BO over 4 feedback gains per surface: {trials} trials x {seeds} seeds")
    with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for surf, best_g, best_V, def_V in pool.imap_unordered(_bo_gains, jobs):
            beat = best_V > def_V + 1e-6
            optima[surf] = {"gains": best_g, "mean_V": best_V, "default_V": def_V,
                            "beats_default": bool(beat)}
            print(f"[{surf:7s}] BO gains mean_V={best_V:+.3f} | default={def_V:+.3f} | "
                  f"{'BETTER (+%.3f) gains=%s' % (best_V - def_V, np.round(best_g, 3).tolist()) if beat else 'default not beaten'}",
                  flush=True)
    with open(GAIN_JSON, "w") as f:
        json.dump(optima, f, indent=2)
    print(f"\nsaved {GAIN_JSON}")


def _compare_job(args):
    from methods.marxefe_optimizer import JointCPG, GAIN_DEFAULT
    JointCPG.ATTITUDE_FEEDBACK = True
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    seed, which = args
    nat = _nat()
    nat.DURATION = COURSE_DURATION
    from methods import terrain
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    box = (bl.numpy(), bu.numpy())
    incumbent = nat.load_incumbent()
    cfg = terrain.sample_natural(seed, reach=COURSE_REACH, band_len=(6.0, 10.0),
                                 start_grass=4.0)
    if which == "noadapt":
        policy = np.asarray(GAIN_DEFAULT, float)
    else:                                              # clairvoyant per-band gains
        # RAMPED switch: a gain is a continuous scalar, so morphing it toward the
        # current band's optimum over ~1 s removes the switch shock that makes an
        # instant jump harmful (instant is worse than no-adapt; ramped beats it).
        gains = {s: np.asarray(v["gains"], float)
                 for s, v in json.load(open(GAIN_JSON)).items()}
        bands = cfg["bands"]
        state = {"cur": np.asarray(GAIN_DEFAULT, float).copy()}
        ramp = float(GAIN_RAMP)
        def policy(pos_y):
            tgt = np.asarray(gains.get(nat._surface_at(bands, pos_y), GAIN_DEFAULT), float)
            step = np.abs(tgt - state["cur"]) / ramp       # linear approach, capped
            state["cur"] = state["cur"] + np.clip(tgt - state["cur"], -step, step)
            return state["cur"]
    res = nat.run_trial(seed, "noadapt", 1.0, incumbent, cfg, box,
                        trigger="cusum", gain_policy=policy)
    return seed, which, int(res["fell"]), float(res["y"][-1]), int(res["reached_end"])


def do_compare(seeds, workers):
    if not os.path.exists(GAIN_JSON):
        raise SystemExit("run `ceiling.py fit` first")
    jobs = [(s, w) for s in range(seeds) for w in ("noadapt", "oracle")]
    ctx = get_context("spawn")
    R = {}
    with ctx.Pool(workers, maxtasksperchild=2) as pool:
        for seed, which, fell, dist, end in pool.imap_unordered(_compare_job, jobs):
            R[(seed, which)] = (fell, dist, end)
    print(f"\nGain ceiling on the natural transect ({seeds} seeds, feedback ON):")
    print(f"  {'arm':16s} {'falls':>8s} {'end':>7s} {'dist[m]':>10s}")
    for w, lab in [("noadapt", "no-adapt (fixed)"), ("oracle", "oracle (per-band)")]:
        f = sum(R[(s, w)][0] for s in range(seeds))
        e = sum(R[(s, w)][2] for s in range(seeds))
        d = np.mean([R[(s, w)][1] for s in range(seeds)])
        print(f"  {lab:16s} {f:3d}/{seeds:<4d} {e:3d}/{seeds:<3d} {d:8.1f}")
    na = np.array([R[(s, "noadapt")][0] for s in range(seeds)])
    orc = np.array([R[(s, "oracle")][0] for s in range(seeds)])
    nad = np.array([R[(s, "noadapt")][1] for s in range(seeds)])
    ord_ = np.array([R[(s, "oracle")][1] for s in range(seeds)])
    saves = int(np.sum(na.astype(bool) & ~orc.astype(bool)))
    loses = int(np.sum(orc.astype(bool) & ~na.astype(bool)))
    from math import comb
    n = saves + loses
    p = (2.0 * sum(comb(n, i) for i in range(min(saves, loses) + 1)) / 2 ** n) if n else 1.0
    print(f"\n  oracle vs no-adapt: saves {saves}, loses {loses} "
          f"(McNemar p={min(p,1.0):.3f}) | dist {(ord_-nad).mean():+.1f}m")
    try:
        from scipy.stats import wilcoxon
        if np.any(ord_ != nad):
            print(f"  distance Wilcoxon p={wilcoxon(ord_, nad)[1]:.3f}")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "compare"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.stage == "fit":
        do_fit(a.trials, max(3, a.seeds if a.seeds < 6 else 3), min(a.workers, 5))
    else:
        do_compare(a.seeds, a.workers)


if __name__ == "__main__":
    main()
