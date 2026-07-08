"""Per-terrain CPG oracle fitting on the 30-s stability criterion.

For a single terrain, run Bayesian optimization over all 8 CPG parameters where
every candidate is scored by ONE 30-s episode with the paper's stability
criterion V (Eq. 7):

    V = min(mean(vx)/v*, 1) - (RMS lp-roll + RMS detrended lp-pitch)[deg]/10,
    V = -2 on a fall (or if the robot never reaches the terrain under test).

N optimization seeds give a DISTRIBUTION of optima (many local optima). This is
the shared engine; per-terrain runners (experiment-flat, experiment-sloped) call
it with their terrain config, and experiment-flat2sloped assembles the cross-
terrain comparison figure. Attitude is in the physical convention (see
methods.episode): pitch = forward tilt (detrended = terrain-relative), roll =
lateral bank (absolute).
"""

import csv
import json
import os
import time
from multiprocessing import get_context

import numpy as np

from methods.episode import (DT, FAR_SLOPE_Y, N_COLS, PARAM_NAMES, TARGET_VX,
                             run_episode)

ATT_REF_DEG = 10.0        # psi_0 attitude normalization [deg]
V_FALL = -2.0             # fall penalty c_f
N_INIT = 8                # random probes before UCB proposals
EPISODE_T = 30.0          # evaluation episode length [s]

PARAM_LABELS = [r"$\gamma$", r"$\omega_{\rm sw}$", r"$\omega_{\rm st}$",
                r"$F_{\rm fast}$", r"$K_{\rm stop}$", r"$A_{\rm hip}$",
                r"$A_{\rm knee}$", r"$b$"]


def _lowpass(x, w=50):
    """0.5-s moving average (removes stride-frequency rocking, keeps drift)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x
    w = min(w, len(x))
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty_like(x)
    for i in range(len(x)):
        a, b = max(0, i - w + 1), i + 1
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def score_V(log, score_from_y=None):
    """Stability criterion V over the episode. `score_from_y` (sloped terrains):
    score from the first step past that forward position; None (flat): score
    after a 1.5-s transient. Never reaching `score_from_y` counts as failure."""
    if log["fell"]:
        return V_FALL
    y = np.asarray(log["y"])
    if score_from_y is not None:
        onto = np.nonzero(y >= score_from_y)[0]
        if len(onto) == 0:
            return V_FALL
        k0 = int(onto[0])
    else:
        k0 = 150
    vx = np.asarray(log["vx"])[k0:]
    roll = np.asarray(log["roll"])[k0:]
    pitch = np.asarray(log["pitch"])[k0:]
    if len(vx) < 200:
        return V_FALL
    r_v = min(max(float(np.mean(vx)), 0.0) / TARGET_VX, 1.0)
    rms_roll = np.rad2deg(np.sqrt(np.mean(_lowpass(roll) ** 2)))          # lateral, absolute
    lp_p = _lowpass(pitch)
    rms_pitch = np.rad2deg(np.sqrt(np.mean((lp_p - np.median(lp_p)) ** 2)))  # forward, detrended
    return r_v - (rms_roll + rms_pitch) / ATT_REF_DEG


# ── Stage 1: per-seed BO ─────────────────────────────────────────────────────

def _limit_threads():
    """Pin BLAS/torch to a single thread per worker. With N worker processes,
    the default multi-threaded BLAS/torch would spawn N x (cores) threads and
    thrash the CPU (~5x slowdown observed); one thread per worker lets N workers
    use N cores cleanly."""
    import os
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _bo_job(job):
    """One optimization seed: BO with 30-s episode evaluations on `cfg`."""
    _limit_threads()
    cfg, score_from_y, seed, n_trials = job
    import torch
    from methods.bo_optimizer import BOOptimizer, BetaSchedule
    from methods.cpg_bounds import bounds_lower, bounds_upper

    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    bo = BOOptimizer(
        bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
        beta_schedule=BetaSchedule(beta_init=5.0, beta_min=1.0,
                                   n_decay_start=max(10, n_trials // 2), gamma=0.9),
        n_init=N_INIT, seed=int(seed))
    rng = np.random.default_rng(30_000 + int(seed))
    t0 = time.time()
    best = dict(V=-np.inf, params=None, fell=None)
    for t in range(n_trials):
        if t < N_INIT:
            x = rng.uniform(lo, hi)
        else:
            try:
                model = bo.fit_model()
                x = bo.from_unit(bo.suggest(model, bo.beta_schedule(t)))
            except Exception:
                x = rng.uniform(lo, hi)
        x = np.clip(np.asarray(x, float), lo, hi)
        log = run_episode(dict(cfg), seed=int(seed), params_start=x, duration=EPISODE_T)
        V = score_V(log, score_from_y)
        bo._append(x, float(V))
        if V > best["V"]:
            best = dict(V=float(V), params=x.tolist(), fell=bool(log["fell"]))
    best.update(seed=int(seed), n_trials=int(n_trials),
                wall_s=round(time.time() - t0, 1))
    return best


def run_oracle(terrain_name, cfg, score_from_y, results_dir, seeds, trials, workers):
    """Fit `seeds` independent BO oracles on the terrain; write oracles.csv."""
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "oracles.csv")
    cols = ["seed", "V", "fell", "n_trials", "wall_s"] + PARAM_NAMES
    jobs = [(cfg, score_from_y, s, trials) for s in range(seeds)]
    ctx = get_context("spawn")
    print(f"[{terrain_name}] {seeds} BO seeds x {trials} trials, 30-s episodes")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        with ctx.Pool(workers, maxtasksperchild=1) as pool:
            for i, r in enumerate(pool.imap_unordered(_bo_job, jobs)):
                row = {k: r[k] for k in ("seed", "V", "fell", "n_trials", "wall_s")}
                row.update({n: round(v, 4) for n, v in zip(PARAM_NAMES, r["params"])})
                w.writerow(row); f.flush()
                print(f"[{i+1:3d}/{seeds}] seed {r['seed']:3d}  V={r['V']:6.3f} "
                      f"fell={int(r['fell'])} ({r['wall_s']:.0f}s)", flush=True)
    print(f"saved {out_csv}")
    return out_csv


# ── Stage 2: pre-select the reference optimum ────────────────────────────────

def _select_job(job):
    _limit_threads()
    cfg, score_from_y, cand_id, params, rep = job
    log = run_episode(dict(cfg), seed=5000 + rep,
                      params_start=np.asarray(params, float), duration=EPISODE_T)
    return cand_id, rep, score_V(log, score_from_y), bool(log["fell"])


def select_oracle(terrain_name, cfg, score_from_y, results_dir, top, reps, workers):
    """Re-evaluate the top candidates over jitter reps; write selected_params.json."""
    rows = list(csv.DictReader(open(os.path.join(results_dir, "oracles.csv"))))
    cands = sorted(rows, key=lambda r: -float(r["V"]))[:top]
    jobs = [(cfg, score_from_y, int(c["seed"]),
             [float(c[n]) for n in PARAM_NAMES], rep)
            for c in cands for rep in range(reps)]
    ctx = get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        res = pool.map(_select_job, jobs)
    print(f"\n[{terrain_name}] top-{top} candidates x {reps} jitter reps:")
    scored = []
    for c in cands:
        cid = int(c["seed"])
        ms = [(V, fe) for (i, _, V, fe) in res if i == cid]
        falls = sum(fe for _, fe in ms)
        mV = float(np.mean([V for V, _ in ms]))
        print(f"  seed {cid:3d}: falls {falls}/{reps}, mean V {mV:6.3f} "
              f"(search V {float(c['V']):6.3f})")
        scored.append((falls, -mV, cid, c))
    falls, negV, cid, c = sorted(scored)[0]
    print(f"  -> selected seed {cid} ({falls}/{reps} falls, mean V {-negV:.3f})")
    sel = {"terrain": terrain_name, "bo_seed": cid, "mean_V": -negV,
           "falls": f"{falls}/{reps}",
           "params": [float(c[n]) for n in PARAM_NAMES]}
    out = os.path.join(results_dir, "selected_params.json")
    with open(out, "w") as f:
        json.dump(sel, f, indent=2)
    print(f"saved {out}")
    return sel


def sloped_cfg(slope_deg=None, slope_start_y=2.0):
    """Sloped terrain config; slope rises at slope_start_y."""
    from methods.episode import SLOPE_DEG
    return {"kind": "sloped", "slope_deg": float(SLOPE_DEG if slope_deg is None
                                                 else slope_deg),
            "slope_start_y": float(slope_start_y), "n_cols": N_COLS}


def flat_cfg():
    """Flat terrain (a sloped heightfield with the ramp out of reach)."""
    return {"kind": "sloped", "slope_deg": 10.0,
            "slope_start_y": FAR_SLOPE_Y, "n_cols": N_COLS}
