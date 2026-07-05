"""Generate per-terrain reference optima θ*(e) for the Track-2 generalization
experiments (RA-L review M6/C2).

For a target terrain (steeper incline, decline, friction drop, …) this:
  1. runs offline Bayesian optimization of the CPG parameters on that terrain
     for several BO seeds — the SAME procedure that produced the existing
     flat / sloped / friction optima (methods.bo_optimizer.bo_optimize_cpg,
     velocity-tracking objective, so the new oracles are directly comparable to
     the ones already used by experiment-eventtrigger);
  2. pre-selects among the per-seed BO winners by re-evaluating each on the
     terrain over several initial-jitter repetitions and keeping the one with
     the fewest falls, then the highest mean objective (identical rule to
     experiment-flat2sloped.run_experiment.preselect);
  3. writes selected_params_<terrain>.json with keys "flat" (the shared flat
     incumbent, copied from selected_params.json) and "oracle" (the new
     terrain optimum), and appends the per-seed candidates to
     figures/cpg_optima_by_parameter.csv for provenance.

Usage (from repo root):
    python experiment-flat2sloped/gen_terrain_optima.py incline15
    python experiment-flat2sloped/gen_terrain_optima.py decline10 --bo-seeds 5 --trials 100

Terrains are named in TERRAINS below; add entries there to cover more.
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_experiment as rx   # reuse run_episode, window_metrics, constants

RESULTS_DIR = os.path.join(_HERE, "results")
OPTIMA_CSV = os.path.join(_REPO, "figures", "cpg_optima_by_parameter.csv")
SELECTED_JSON = os.path.join(RESULTS_DIR, "selected_params.json")

TARGET_VX = 0.5
ROBOT_MASS = 10.0
PRESELECT_REPS = 8          # jitter reps per candidate in the pre-selection
PRESELECT_T = 10.0          # seconds evaluated per candidate


def _sloped_eval_cfg(slope_deg):
    """Terrain used to BOTH optimize and pre-select on a sloped terrain: the
    ramp starts at 2 m so the robot climbs (or descends) for most of the bout —
    identical to experiment-flat2sloped.preselect's sloped candidates."""
    return {"kind": "sloped", "slope_deg": float(slope_deg),
            "slope_start_y": 2.0, "n_cols": rx.N_COLS}


# name -> dict(bo_cfg, eval_cfg, csv_terrain). bo_cfg is the terrain set on
# terrain.TERRAIN_CONFIG for the offline BO; eval_cfg is used for pre-selection.
TERRAINS = {
    "incline15": dict(bo_cfg=_sloped_eval_cfg(15.0),
                      eval_cfg=_sloped_eval_cfg(15.0), csv_terrain="incline15"),
    "incline20": dict(bo_cfg=_sloped_eval_cfg(20.0),
                      eval_cfg=_sloped_eval_cfg(20.0), csv_terrain="incline20"),
    "decline10": dict(bo_cfg=_sloped_eval_cfg(-10.0),
                      eval_cfg=_sloped_eval_cfg(-10.0), csv_terrain="decline10"),
}


# ── Stage 1: offline BO per seed (one subprocess-free worker per seed) ────────

def _bo_job(job):
    terrain_name, seed, n_trials = job
    from methods import terrain
    from methods.bo_optimizer import bo_optimize_cpg
    from methods.cpg_bounds import bounds

    spec = TERRAINS[terrain_name]
    terrain.TERRAIN_CONFIG = dict(spec["bo_cfg"])
    tmp_dir = os.path.join(RESULTS_DIR, f"bo_{terrain_name}")
    os.makedirs(tmp_dir, exist_ok=True)
    out = bo_optimize_cpg(
        bounds, target_velocity=TARGET_VX, robot_mass=ROBOT_MASS,
        n_trials=n_trials, n_init=5, optimizer_name=f"BO_{terrain_name}",
        seed=int(seed), results_dir=tmp_dir)
    best_params = np.asarray(out[2], float)   # (train_X, train_Y, best_params, ...)
    return terrain_name, int(seed), best_params.tolist()


# ── Stage 2: pre-selection (re-evaluate each candidate on the terrain) ────────

def _preselect_job(job):
    terrain_name, cand_seed, params, rep = job
    spec = TERRAINS[terrain_name]
    log = rx.run_episode(dict(spec["eval_cfg"]), seed=3000 + rep,
                         params_start=np.asarray(params, float),
                         duration=PRESELECT_T)
    m = rx.window_metrics(log, 0, int(PRESELECT_T / rx.DT))
    m["fell"] = bool(log["fell"])
    if m["fell"]:
        m["J"] = -50.0
    return terrain_name, cand_seed, rep, m


def generate(terrain_name, bo_seeds, n_trials, workers):
    if terrain_name not in TERRAINS:
        raise SystemExit(f"unknown terrain {terrain_name!r}; "
                         f"choices: {list(TERRAINS)}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ctx = get_context("spawn")

    # Stage 1: BO candidates
    print(f"[{terrain_name}] offline BO: {bo_seeds} seeds x {n_trials} trials")
    bo_jobs = [(terrain_name, s, n_trials) for s in range(bo_seeds)]
    with ctx.Pool(min(workers, bo_seeds), maxtasksperchild=1) as pool:
        cands = pool.map(_bo_job, bo_jobs)
    cands = [(cs, np.asarray(v, float)) for (_, cs, v) in cands]
    for cs, v in cands:
        print(f"  BO seed {cs}: {np.round(v, 3).tolist()}")

    # Stage 2: pre-selection over jitter reps
    print(f"[{terrain_name}] pre-selecting over {PRESELECT_REPS} jitter reps")
    ps_jobs = [(terrain_name, cs, v, rep)
               for (cs, v) in cands for rep in range(PRESELECT_REPS)]
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        ps = pool.map(_preselect_job, ps_jobs)

    scores = []
    for cs, v in cands:
        ms = [m for (_, c, _, m) in ps if c == cs]
        falls = sum(m["fell"] for m in ms)
        J = float(np.mean([m["J"] for m in ms]))
        dist = float(np.mean([m["dist"] for m in ms]))
        print(f"  BO seed {cs}: falls {falls}/{PRESELECT_REPS}, "
              f"mean dist {dist:5.2f} m, mean J {J:6.2f}")
        scores.append((falls, -J, cs, v))
    falls, negJ, best_cs, best_v = sorted(scores, key=lambda t: (t[0], t[1]))[0]
    print(f"  -> selected BO seed {best_cs} ({falls}/{PRESELECT_REPS} falls, "
          f"J = {-negJ:.2f})")

    # Provenance: append candidates to the shared optima CSV.
    _append_optima_csv(TERRAINS[terrain_name]["csv_terrain"], cands)

    # Output JSON: flat incumbent (shared) + this terrain's oracle.
    with open(SELECTED_JSON) as f:
        flat = json.load(f)["flat"]
    out = {"flat": flat,
           "oracle": {"bo_seed": int(best_cs), "params": best_v.tolist()},
           "terrain": terrain_name}
    out_path = os.path.join(RESULTS_DIR, f"selected_params_{terrain_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


def _append_optima_csv(csv_terrain, cands):
    cols = ["terrain", "seed", "coupling_gain", "w_swing", "w_stance",
            "F_FAST", "STOP_GAIN", "hip_amp", "knee_amp", "b"]
    # Idempotent: drop any existing rows for this terrain so re-running
    # regenerates cleanly instead of duplicating.
    rows = []
    if os.path.exists(OPTIMA_CSV):
        with open(OPTIMA_CSV) as f:
            r = csv.reader(f)
            header = next(r, None)
            rows = [row for row in r if row and row[0] != csv_terrain]
    with open(OPTIMA_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
        for cs, v in cands:
            w.writerow([csv_terrain, int(cs)] + [round(float(x), 4) for x in v])
    print(f"wrote {len(cands)} {csv_terrain} rows to {OPTIMA_CSV} "
          f"(kept {len(rows)} other rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("terrain", choices=list(TERRAINS))
    ap.add_argument("--bo-seeds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    generate(args.terrain, args.bo_seeds, args.trials, args.workers)


if __name__ == "__main__":
    main()
