"""Natural-landscape online CPG adaptation, triggered by prediction error.

This applies the *same* triggered / squash-to-stop online-adaptation protocol as
`experiment-flat2slope-adapt` (single flat->slope change), but to a much harder
scenario: a NATURAL transect with MANY terrain changes. The robot walks forward
across a random sequence of forward bands -- grass / gravel / rocks / river --
each with its own friction AND its own geometry (grass: gentle undulation;
gravel: fine roughness; rocks: scattered low bumps to step over; river: a
slippery depression). See `methods.terrain.sample_natural`.

Because the optimum shifts band by band there is no single "oracle" gait, so the
oracle anchor of the flat->slope experiment is dropped. The remaining arms are:

  * noadapt  -> hold the flat/grass-optimal gait for the whole transect (anchor);
  * grid     -> Latin-hypercube search, one window at a time (safeguarded);
  * bo       -> GP-UCB on the windowed stability objective (safeguarded);
  * marxefe  -> active-inference EFE selection, model updated every step.

The MARX-EFE goal-cross-entropy monitor (or the decision-theoretic Newton
decrement, `--trigger dt`) runs every step. Normalised to a grass-prefix baseline
it is ~1 on the opening grass and rises at each band transition; each rising edge
FIRES an event (the monitor re-arms once it falls back) so the trigger fires
repeatedly, once per transition. SQUASH: a method adapts only while its error is
above K, and pauses once a full window's mean ratio drops back below K -- so it
re-adapts at every band and rests in between. A shared revert-to-best safeguard
protects the destructive optimisers.

Unlike the single-change experiment there is no fixed post-trigger horizon: each
run is one long continuous bout over the whole transect, ending on a fall, at the
duration cap, or when the robot reaches the end of the sampled terrain. Survival
distance is therefore the discriminating metric.

The monitors, online methods, safeguard and windowed objective are imported
unchanged from `experiment-flat2slope-adapt/run_experiment.py` (single source of
truth); this module only swaps the terrain, the per-step friction update, the
(terrain-relative) fall check, the continuous multi-transition run loop, and the
survival-distance metrics.

Usage (from repo root):
    python experiment-natural-adapt/run_experiment.py run \
        --seeds 10 --arms noadapt grid bo marxefe --workers 10
    # then: python experiment-natural-adapt/analyze.py
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import time as _time
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load_f2s():
    """Load experiment-flat2slope-adapt/run_experiment.py as a module (its
    basename collides with this file, so load it explicitly under a new name).
    Importing is side-effect-free: its argparse lives under __main__."""
    path = os.path.join(_REPO, "experiment-flat2slope-adapt", "run_experiment.py")
    spec = importlib.util.spec_from_file_location("f2s_adapt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f2s = _load_f2s()

# Reused, unchanged from the flat->slope experiment (single source of truth):
DT = f2s.DT
WINDOW = f2s.WINDOW
RAMP = f2s.RAMP
TARGET_VX = f2s.TARGET_VX
DEFAULT_ORI = f2s.DEFAULT_ORI
LEG_NAMES = f2s.LEG_NAMES
J_FALL = f2s.J_FALL
J_GOOD = f2s.J_GOOD
BASELINE_T = f2s.BASELINE_T
ADAPT_T = f2s.ADAPT_T                       # kept only for the safeguard warm score
TriggerMonitor = f2s.TriggerMonitor
DecisionTheoreticMonitor = f2s.DecisionTheoreticMonitor
CusumDecisionMonitor = f2s.CusumDecisionMonitor
Safeguard = f2s.Safeguard
j_stab_window = f2s.j_stab_window
_reset_with_jitter = f2s._reset_with_jitter
CONTROL_PRIOR_SCALE = f2s.CONTROL_PRIOR_SCALE
DT_BUDGET_MOVE = f2s.DT_BUDGET_MOVE

# Natural online arms: drop the (ill-defined) oracle; keep the searchers + anchor.
METHODS = {n: f2s.METHODS[n] for n in ("noadapt", "grid", "bo", "marxefe")}
# `oracle` is not a searcher (it has no method object); it is the clairvoyant
# per-band reference handled directly in run_trial. It is a valid arm but is kept
# out of METHODS (which maps arm -> searcher class).
ARMS_DEFAULT = ["noadapt", "grid", "bo", "marxefe"]
ALL_ARMS = ARMS_DEFAULT + ["oracle"]

RESULTS_DIR = os.path.join(_HERE, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
MANIFEST_CSV = os.path.join(RESULTS_DIR, "manifest.csv")
CONFIG_JSON = os.path.join(RESULTS_DIR, "config.json")

# The incumbent (starting) gait is the flat-optimal one: the transect opens on
# grass (mu~0.70), essentially flat, so the flat optimum is the natural anchor.
FLAT_JSON = os.path.join(_REPO, "experiment-flat", "results", "selected_params.json")

# ── Natural terrain / episode ────────────────────────────────────────────────
# Long bands: each friction/geometry regime persists ~12-20 s of walking (at
# v*=0.5 m/s), so a transition is a sustained mismatch the DT trigger can lock
# onto and a method has time to adapt within a band before the next one.
DURATION = 80.0            # hard cap on the continuous bout [s]
REACH = 45.0             # forward extent of the sampled transect [m]
BAND_LEN = (6.0, 10.0)    # per-band length range [m]
START_GRASS = 4.0         # opening grass run before the first transition [m]
CLEAR_FALL = 0.22         # base clearance above local ground below which = fall [m]
UPRIGHT_FALL = 0.30       # world-up . body-up below which we call a tip-over
END_MARGIN = 1.0          # stop as "reached end" this far before REACH [m]


def load_surface_optima():
    """{surface -> 8-vector} of the per-band CPG optima fit by fit_surface_oracles.py.
    Falls back to the incumbent for any surface without an entry."""
    import json as _json
    path = os.path.join(RESULTS_DIR, "surface_optima.json")
    if not os.path.exists(path):
        raise SystemExit("surface_optima.json not found; run fit_surface_oracles.py "
                         "before the oracle arm")
    d = _json.load(open(path))
    return {surf: np.asarray(v["params"], float) for surf, v in d.items()}


def _surface_at(bands, y):
    """Surface name of the band the robot is currently on (forward position y)."""
    nm = bands[0][1]
    for yb, b in bands:
        if y >= yb:
            nm = b
        else:
            break
    return str(nm)


def load_incumbent():
    with open(FLAT_JSON) as f:
        return np.asarray(json.load(f)["params"], float)


def _fallen_natural(base_pos, base_ori, p, cfg):
    """Natural-terrain fall check. With per-band inclines the robot climbs, so an
    absolute base-height floor is invalid; test the base clearance above the LOCAL
    ground elevation (terrain.natural_elev_at) instead, plus a tip-over test on the
    body-up axis."""
    from methods import terrain
    rot = p.getMatrixFromQuaternion(base_ori)
    upright = float(np.dot([0, 0, 1], rot[6:]))
    clear = base_pos[2] - terrain.natural_elev_at(cfg, base_pos[1])
    return (upright < UPRIGHT_FALL or clear < CLEAR_FALL), upright


# ── One monitored, squash-adaptive, continuous natural bout ──────────────────

def run_trial(seed, method_name, k_sigma, incumbent, cfg, box, trigger="ce",
              gain_policy=None):
    """One continuous bout over a natural transect. Same trigger / squash /
    safeguard machinery as the flat->slope experiment, but the run does not stop
    a fixed horizon after the first trigger: it continues over every band,
    re-triggering at each transition, until a fall, the duration cap, or the end
    of the terrain. Returns the full per-step signals for NPZ dumping.

    `gain_policy` (optional) adapts the CPG's attitude-feedback gains, the
    continuous alternative to switching the gait-shape parameters: a fixed
    4-vector [kp_roll, kd_roll, kp_pitch, kd_pitch] applied throughout, or a
    callable(pos_y) -> 4-vector applied each step (the clairvoyant per-band gain
    oracle). The 8-D CPG shape parameters are still driven by `method_name`."""
    import pybullet as p
    from methods import terrain
    from methods.marxefe_optimizer import (get_base_orientation,
                                           load_environment, load_robot)

    terrain.TERRAIN_CONFIG = cfg
    load_environment(DT, use_gui=False)
    robot, _, joint_IDs_full, _, feet = load_robot(p)
    cpg = _reset_with_jitter(p, robot, seed)
    if gain_policy is not None and not callable(gain_policy):
        cpg.set_gains(gain_policy)                  # fixed feedback gains

    n_steps = int(round(DURATION / DT))
    Monitor = {"dt": DecisionTheoreticMonitor, "cusum": CusumDecisionMonitor}.get(
        trigger, TriggerMonitor)
    monitor = Monitor(n_steps, k_sigma)
    # The clairvoyant `oracle` arm switches, band by band, to the pre-fit optimum
    # for the surface the robot is currently on (no trigger, no search): an upper
    # bound on what any parameter switch could achieve. Other arms use the trigger.
    is_oracle = (method_name == "oracle")
    surf_optima = load_surface_optima() if is_oracle else None
    method = None if is_oracle else METHODS[method_name](
        np.asarray(incumbent, float), np.asarray(incumbent, float), box, seed)

    keys = ["t", "x", "y", "z", "vx", "vy", "roll", "pitch", "upright"]
    log = {k: np.zeros(n_steps) for k in keys}
    applied_log = np.zeros((8, n_steps))
    adapting_log = np.zeros(n_steps, dtype=int)

    seg_start = np.asarray(incumbent, float).copy()
    seg_target = seg_start.copy()
    seg_anchor = 0
    applied = seg_start.copy()
    pos_y = 0.0                                    # forward position for friction
    roll = pitch = 0.0                             # previous-step attitude for CPG feedback

    trigger_step = None
    guard = None
    window_scores = []
    selected_params = []
    win_buf = {"vx": [], "roll": [], "pitch": []}
    fell, fall_step = False, None
    reached_end = False
    adapting = True
    n_fires_seen = 0
    n_pauses = 0
    adapt_windows = 0
    propose_times = []

    for k in range(n_steps):
        t = k * DT

        # ORACLE: clairvoyantly switch to the current band's surface optimum
        # (ramped) whenever the robot crosses into a new band. No trigger/search.
        if is_oracle:
            surf = _surface_at(cfg["bands"], pos_y)
            tgt = np.asarray(surf_optima.get(surf, incumbent), float)
            if not np.array_equal(tgt, seg_target):
                seg_start = applied.copy()
                seg_target = tgt
                seg_anchor = k

        # Window boundary (post first trigger): score the finished window, ask
        # the method for the next candidate, ramp toward it. SQUASH: pause once
        # the ratio is back below K; resume on a re-fire of the monitor.
        if (not is_oracle and trigger_step is not None and k > trigger_step
                and (k - trigger_step) % WINDOW == 0):
            last_J = j_stab_window(win_buf["vx"], win_buf["roll"],
                                   win_buf["pitch"], fell=False)
            window_scores.append(last_J)
            win_buf = {"vx": [], "roll": [], "pitch": []}
            win_ratio = (np.mean(monitor.ema_log[max(0, k - WINDOW):k])
                         / monitor._baseline_mean())
            if adapting and win_ratio < k_sigma and adapt_windows >= 2:
                adapting = False
                n_pauses += 1
            elif not adapting and len(monitor.fire_steps) > n_fires_seen:
                adapting = True
            n_fires_seen = len(monitor.fire_steps)
            if adapting:
                t0 = _time.perf_counter()
                target = guard.next_target(last_J)
                propose_times.append(_time.perf_counter() - t0)
                adapt_windows += 1
                if target is not None:
                    seg_start = applied.copy()
                    seg_target = np.asarray(target, float)
                    seg_anchor = k
            selected_params.append(np.asarray(seg_target, float).copy())

        frac = min(1.0, (k - seg_anchor) / max(1, RAMP))
        applied = seg_start + frac * (seg_target - seg_start)

        # Set the ground friction to the band the robot is currently standing on.
        terrain.apply_dynamic_friction(p, robot, pos_y)
        if callable(gain_policy):                   # clairvoyant per-band gains
            cpg.set_gains(gain_policy(pos_y))

        raw = np.array([int(len(p.getContactPoints(
            bodyA=0, bodyB=robot, linkIndexA=-1, linkIndexB=feet[j])) > 0)
            for j in range(4)])
        hips, knees = cpg.step(applied, raw, DT, roll=roll, pitch=pitch)
        for j in range(4):
            a_id, h_id, k_id = joint_IDs_full[LEG_NAMES[j]]
            p.setJointMotorControl2(robot, a_id, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j])
            p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()

        base_pos, base_ori = get_base_orientation(p, robot, DEFAULT_ORI)
        vel, _ = p.getBaseVelocity(robot)
        pitch, roll, _ = p.getEulerFromQuaternion(base_ori)  # physical (+Y fwd)
        pos_y = base_pos[1]
        fallen, upright = _fallen_natural(base_pos, base_ori, p, cfg)

        log["t"][k] = t
        log["x"][k], log["y"][k], log["z"][k] = base_pos[0], base_pos[1], base_pos[2]
        log["vx"][k], log["vy"][k] = vel[1], vel[0]
        log["roll"][k], log["pitch"][k], log["upright"][k] = roll, pitch, upright
        applied_log[:, k] = applied
        adapting_log[k] = int(adapting)

        y_new = np.array([vel[1], vel[0], pitch, roll])
        fired = monitor.step(k, t, y_new, applied)
        if method is not None:
            method.on_step(y_new, applied)

        if not is_oracle and fired and trigger_step is None:
            trigger_step = monitor.fire_step
            k0 = max(0, k - WINDOW)
            pre_J = j_stab_window(log["vx"][k0:k + 1], log["roll"][k0:k + 1],
                                  log["pitch"][k0:k + 1], fell=False)
            guard = Safeguard(method, incumbent, pre_J)
            t0 = _time.perf_counter()
            target = guard.next_target(None)
            propose_times.append(_time.perf_counter() - t0)
            adapt_windows += 1
            n_fires_seen = len(monitor.fire_steps)
            if target is not None:
                seg_start = applied.copy()
                seg_target = np.asarray(target, float)
                seg_anchor = k + 1
            selected_params.append(np.asarray(seg_target, float).copy())
            win_buf = {"vx": [], "roll": [], "pitch": []}
        elif trigger_step is not None:
            win_buf["vx"].append(vel[1])
            win_buf["roll"].append(roll)
            win_buf["pitch"].append(pitch)

        if fallen:
            fell, fall_step = True, k
            if trigger_step is not None:
                window_scores.append(J_FALL)
            n_steps = k + 1
            break
        if pos_y >= cfg["reach"] - END_MARGIN:     # walked the whole transect
            reached_end = True
            n_steps = k + 1
            break

    p.disconnect()

    n = n_steps
    for kk in keys:
        log[kk] = log[kk][:n]
    applied_log = applied_log[:, :n]
    adapting_log = adapting_log[:n]

    return dict(
        # per-step signals
        t=log["t"], x=log["x"], y=log["y"], z=log["z"],
        vx=log["vx"], vy=log["vy"], roll=log["roll"], pitch=log["pitch"],
        upright=log["upright"], applied=applied_log, adapting=adapting_log,
        ce_raw=monitor.c_log[:n], ce_ema=monitor.ema_log[:n],
        ratio=monitor.ratio_trace()[:n],
        dt_ctrl=(monitor.ctrl_log[:n] if trigger in ("dt", "cusum") else np.zeros(n)),
        cusum_s=(monitor.s_log[:n] if trigger == "cusum" else np.zeros(n)),
        trigger_kind=trigger,
        # events / summaries
        window_scores=np.asarray(window_scores, float),
        selected_params=np.asarray(selected_params, float) if selected_params
        else np.zeros((0, 8)),
        fire_steps=np.asarray([s for s in monitor.fire_steps if s < n], int),
        trigger_step=(-1 if trigger_step is None else int(trigger_step)),
        fall_step=(-1 if fall_step is None else int(fall_step)),
        fell=int(fell), reached_end=int(reached_end),
        adapt_windows=int(adapt_windows), n_pauses=int(n_pauses),
        baseline_mean=float(monitor._baseline_mean()),
        # terrain identity
        bands_y=np.asarray([b[0] for b in cfg["bands"]], float),
        bands_name=np.asarray([b[1] for b in cfg["bands"]]),
        zones_mu=np.asarray([z[1] for z in cfg["zones"]], float),
        reach=float(cfg["reach"]), base_mu=float(cfg["base_mu"]),
        # run identity / config
        seed=int(seed), method=method_name, k_sigma=float(k_sigma),
        incumbent=np.asarray(incumbent, float), dt=float(DT),
        window=int(WINDOW), ramp=int(RAMP), duration=float(DURATION),
        propose_t_total=float(np.sum(propose_times)) if propose_times else 0.0,
    )


# ── Scalar metrics for the manifest ──────────────────────────────────────────

def _bands_crossed(bands_y, final_y):
    """How many band transitions the robot walked over (excludes the start band)."""
    return int(np.sum((bands_y > 0.0) & (bands_y <= final_y)))


def _tip_dev_deg(roll, pitch):
    return np.rad2deg(np.sqrt(np.asarray(roll) ** 2 + np.asarray(pitch) ** 2))


def scalar_metrics(res):
    n = len(res["y"])
    kT = res["trigger_step"]
    triggered = int(kT >= 0)
    fell = int(res["fell"])
    fall_step = res["fall_step"]
    final_y = float(res["y"][-1]) if n else 0.0
    t = res["t"]
    mtail = np.asarray(t) >= BASELINE_T[1]         # after the baseline window
    tip = _tip_dev_deg(res["roll"], res["pitch"])
    ws = res["window_scores"]
    good = [i for i, J in enumerate(ws) if J >= J_GOOD]
    d = dict(
        seed=res["seed"], method=res["method"],
        triggered=triggered, fell=fell, reached_end=int(res["reached_end"]),
        dist=final_y,                              # survival distance (key metric)
        t_end=float(t[-1]) if n else 0.0,
        bands_crossed=_bands_crossed(res["bands_y"], final_y),
        fall_t=(fall_step * DT if fall_step >= 0 else np.nan),
        trigger_t=(kT * DT if triggered else np.nan),
        n_triggers=int(len(res["fire_steps"])),
        n_proposals=int(res["adapt_windows"]),
        n_pauses=int(res["n_pauses"]),
        n_windows=int(len(ws)),
        mean_J=(float(np.mean(ws)) if len(ws) else np.nan),
        win_to_good=(int(good[0]) if good else -1),
        mean_tip=(float(np.mean(tip[mtail])) if mtail.any() else np.nan),
        max_tip=(float(np.max(tip[mtail])) if mtail.any() else np.nan),
        mean_vx=(float(np.mean(np.asarray(res["vx"])[mtail])) if mtail.any() else np.nan),
        baseline_ce=res["baseline_mean"],
    )
    return d


def run_path(seed, method, trigger="ce"):
    suffix = "" if trigger == "ce" else f"_{trigger}"
    return os.path.join(RUNS_DIR, f"nat_seed{seed}_{method}{suffix}.npz")


# ── Job / harness ────────────────────────────────────────────────────────────

def _job(args):
    f2s._limit_threads()
    seed, method, k_sigma, incumbent, trigger, dt_move, cusum_slack, cusum_h = args
    f2s.DT_BUDGET_MOVE = float(dt_move)            # picked up by the DT monitor
    if cusum_slack is not None:
        f2s.DT_CUSUM_SLACK = float(cusum_slack)   # picked up by the CUSUM monitor
    if cusum_h is not None:
        f2s.DT_CUSUM_H = float(cusum_h)
    from methods.marxefe_optimizer import JointCPG   # attitude-feedback ablation
    JointCPG.ATTITUDE_FEEDBACK = os.environ.get("CPG_ATTITUDE_FB", "1") != "0"
    from methods import terrain
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    box = (bl.numpy(), bu.numpy())
    cfg = terrain.sample_natural(seed, reach=REACH, band_len=BAND_LEN,
                                 start_grass=START_GRASS)
    res = run_trial(seed, method, k_sigma, incumbent, cfg, box, trigger=trigger)
    path = run_path(seed, method, trigger)
    np.savez_compressed(path, **res)
    row = scalar_metrics(res)
    row["trigger"] = trigger
    row["npz"] = os.path.relpath(path, RESULTS_DIR)
    return row


def run(seeds, arms, k_sigma, workers, trigger="ce", dt_move=DT_BUDGET_MOVE,
        cusum_slack=None, cusum_h=None):
    os.makedirs(RUNS_DIR, exist_ok=True)
    incumbent = load_incumbent()
    k_eff = 1.0 if trigger in ("dt", "cusum") else float(k_sigma)
    tau = 8 * (4.0 * dt_move / CONTROL_PRIOR_SCALE) ** 2
    suffix = "" if trigger == "ce" else f"_{trigger}"
    manifest = MANIFEST_CSV if trigger == "ce" else os.path.join(
        RESULTS_DIR, f"manifest{suffix}.csv")
    config = CONFIG_JSON if trigger == "ce" else os.path.join(
        RESULTS_DIR, f"config{suffix}.json")

    print(f"incumbent (flat/grass-optimal): {np.round(incumbent, 3).tolist()}")
    cs = f2s.DT_CUSUM_SLACK if cusum_slack is None else cusum_slack
    ch = f2s.DT_CUSUM_H if cusum_h is None else cusum_h
    print(f"trigger={trigger}  seeds={seeds} arms={arms} threshold={k_eff}"
          + (f"  tau={tau:.3f} (move={dt_move})" if trigger in ("dt", "cusum") else "")
          + (f"  cusum kappa={cs} h={ch}" if trigger == "cusum" else ""))
    print(f"natural transect: reach={REACH} m, bands {BAND_LEN} m, "
          f"grass start {START_GRASS} m; bout cap {DURATION} s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(config, "w") as f:
        json.dump(dict(trigger=trigger, seeds=seeds, arms=arms, k_sigma=k_eff,
                       dt_move=dt_move, tau=tau,
                       control_prior_scale=CONTROL_PRIOR_SCALE,
                       reach=REACH, band_len=list(BAND_LEN),
                       start_grass=START_GRASS, dt=DT, window=WINDOW, ramp=RAMP,
                       duration=DURATION, target_vx=TARGET_VX,
                       incumbent=incumbent.tolist()), f, indent=2)

    jobs = [(int(s), m, k_eff, incumbent, trigger, dt_move, cusum_slack, cusum_h)
            for s in range(seeds) for m in arms]
    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=2) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(row)
            tg = "trig" if row["triggered"] else "NO-TRIG"
            end = ("END" if row["reached_end"] else ("FELL" if row["fell"] else "cap"))
            print(f"[{i+1:3d}/{len(jobs)}] seed{row['seed']:>2} "
                  f"{row['method']:<8} {tg:>7} {end:>4}  "
                  f"dist={row['dist']:.1f}m bands={row['bands_crossed']} "
                  f"trig={row['n_triggers']} meanJ={row.get('mean_J', float('nan')):.2f}",
                  flush=True)

    rows.sort(key=lambda r: (r["seed"], r["method"]))
    cols = ["seed", "method", "trigger", "triggered", "fell", "reached_end",
            "dist", "t_end", "bands_crossed", "trigger_t", "fall_t",
            "n_triggers", "n_proposals", "n_pauses", "n_windows", "mean_J",
            "win_to_good", "mean_tip", "max_tip", "mean_vx", "baseline_ce", "npz"]
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nsaved {manifest}  ({len(rows)} runs) and per-run NPZ in {RUNS_DIR}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=ARMS_DEFAULT,
                    choices=ALL_ARMS)
    ap.add_argument("--K", type=float, default=f2s.K_DEFAULT,
                    help="CE trigger threshold on the grass-baseline-normalised ratio")
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="ce",
                    help="ce: goal cross-entropy ratio; dt: decision-theoretic "
                         "Newton-decrement lambda^2 vs a control-cost budget; "
                         "cusum: CUSUM accumulation of lambda^2 (fires on sustained "
                         "suboptimality, e.g. a friction transition)")
    ap.add_argument("--dt-move", type=float, default=DT_BUDGET_MOVE,
                    help="dt/cusum: reference gait move (fraction of param range)")
    ap.add_argument("--cusum-slack", type=float, default=None,
                    help="cusum only: tolerated per-step suboptimality kappa (tau units)")
    ap.add_argument("--cusum-h", type=float, default=None,
                    help="cusum only: CUSUM decision bound h (tau units)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-attitude-fb", action="store_true",
                    help="disable the CPG's VMC body-attitude feedback (ablation)")
    args = ap.parse_args()
    os.environ["CPG_ATTITUDE_FB"] = "0" if args.no_attitude_fb else "1"
    run(args.seeds, args.arms, args.K, args.workers,
        trigger=args.trigger, dt_move=args.dt_move,
        cusum_slack=args.cusum_slack, cusum_h=args.cusum_h)


if __name__ == "__main__":
    main()
