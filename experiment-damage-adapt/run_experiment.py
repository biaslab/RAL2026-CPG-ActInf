"""Actuator-degradation (leg-damage) online CPG adaptation, triggered by
prediction error.

Same triggered / squash-to-stop online-adaptation protocol as
`experiment-flat2slope-adapt` and `experiment-payload-adapt`, but the change is a
partial ACTUATOR FAILURE in ONE leg: the Laikago walks on a flat plane and
halfway through the bout the hip + knee motors of a single (hind) leg lose most
of their torque budget -- their POSITION_CONTROL maxForce is ramped down from
`healthy_force` to `damage_force` over ~1 s. This is the canonical Cully et al.
(Nature 2015) leg-damage setting, the textbook demonstration that adaptation
beats a fixed policy.

Unlike a slope or friction edge, this is a PERSISTENT mismatch: the weakened leg
under-tracks its CPG targets on every stride, dragging and destabilising the
gait until it is re-tuned, so suboptimality accumulates instead of being a
one-off transition shock. The degradation is deliberately ramped (over ramp_t)
so the discriminating signal is the sustained mismatch, not an impulse.

CAVEAT (see notes/adaptation-challenge-candidates.md #3): the 8 CPG parameters
are GLOBAL / symmetric across legs, so the controller cannot simply command more
torque to the weak leg. Compensation has to come indirectly -- a slower gait
(lower w_swing/w_stance gives the weak motor more time to reach its target under
the reduced torque), smaller amplitudes (less torque demanded), higher STOP_GAIN,
re-tuned attitude gains. Whether the parameterisation can express a compensating
gait AT ALL is exactly what `fit_damage_oracles.py` screens: if the damaged-leg
oracle is no better than no-adapt, the parameterisation -- not the damage -- is
the bottleneck. (Screen 2026-07: at 22 Nm on a hind leg it CAN -- a low-amplitude
gait recovers where the incumbent falls; see README.)

Phases:
  phase 1 (t < t_damage): all legs healthy -- the mild condition the incumbent
    handles well;
  phase 2 (t >= t_damage): one leg's hip+knee weakened -- persistent drag /
    roll-pitch bias, the condition that should require re-tuning.

Arms (same METHODS objects as flat2slope, single source of truth):
  * noadapt  -> hold the flat-optimal gait throughout (anchor);
  * grid     -> Latin-hypercube search, one window at a time (safeguarded);
  * bo       -> GP-UCB on the windowed stability objective (safeguarded);
  * marxefe  -> active-inference EFE selection, model updated every step and
                TRAINED during phase 1 by a smooth OU parameter excitation
                (--no-train ablates);
  * oracle   -> clairvoyant: per-phase pre-fit optimum (fit_damage_oracles.py),
                switched exactly at the damage onset; upper-bounds any param switch.

The monitor (CE ratio / DT / CUSUM) runs every step; its baseline window
(2.4-3.4 s) lies inside phase 1, so the trigger detects the DAMAGE, not the
healthy gait. SQUASH pauses adaptation once the error ratio falls back below K.

Besides falls, per-step actuator mechanical power is logged: with a weak leg,
"no-adapt limps along inefficiently" is a real outcome fall counts alone miss.

Usage (from repo root):
    python experiment-damage-adapt/run_experiment.py run \
        --seeds 20 --arms noadapt grid bo marxefe --workers 10
    # oracle arm additionally needs: python experiment-damage-adapt/fit_damage_oracles.py
    # then: python experiment-damage-adapt/analyze.py
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import warnings
import time as _time
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load_f2s():
    """Load experiment-flat2slope-adapt/run_experiment.py as a module (its
    basename collides with this file, so load it explicitly under a new name)."""
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
LEG_NAMES = f2s.LEG_NAMES                 # ["FL", "FR", "RL", "RR"]
J_FALL = f2s.J_FALL
J_GOOD = f2s.J_GOOD
BASELINE_T = f2s.BASELINE_T
TriggerMonitor = f2s.TriggerMonitor
DecisionTheoreticMonitor = f2s.DecisionTheoreticMonitor
CusumDecisionMonitor = f2s.CusumDecisionMonitor
Safeguard = f2s.Safeguard
j_stab_window = f2s.j_stab_window
_reset_with_jitter = f2s._reset_with_jitter
CONTROL_PRIOR_SCALE = f2s.CONTROL_PRIOR_SCALE
DT_BUDGET_MOVE = f2s.DT_BUDGET_MOVE

METHODS = {n: f2s.METHODS[n] for n in ("noadapt", "grid", "bo", "marxefe")}
ARMS_DEFAULT = ["noadapt", "grid", "bo", "marxefe"]
ALL_ARMS = ARMS_DEFAULT + ["oracle"]

# The MARX-EFE agent is TRAINED in this experiment (like the BO agent): a smooth
# OU excitation on the 8 CPG parameters while walking safely in phase 1 lets the
# MARX model see VARIED inputs and identify the u -> y map before the damage
# (with constant u the input coefficients are unidentified and the EFE is flat in
# u). Disable with --no-train for the untrained ablation.
MARX_PRIOR_SCALE = 0.5     # control-prior width: trust region for EFE moves
MARX_VEL_STD = 0.2         # goal-prior std on vx,vy [m/s]: makes the post-damage
                           # velocity deficit visible to the EFE
MARX_FORGETTING = 0.999    # model memory ~10 s: training data survives to the damage

RESULTS_DIR = os.path.join(_HERE, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
MANIFEST_CSV = os.path.join(RESULTS_DIR, "manifest.csv")
CONFIG_JSON = os.path.join(RESULTS_DIR, "config.json")
OPTIMA_JSON = os.path.join(RESULTS_DIR, "damage_optima.json")

FLAT_JSON = os.path.join(_REPO, "experiment-flat", "results", "selected_params.json")

# ── Leg-damage scenario defaults ─────────────────────────────────────────────
# Screened 2026-07 (see README): the hip+knee motors of ONE leg are ramped from
# `healthy_force` (60 Nm ~ the uncapped/default behaviour; peak walking torque is
# well under this so 60 Nm is indistinguishable from default) down to
# `damage_force`. A FRONT leg is trivially absorbed (support, not propulsion);
# a HIND leg is the propulsion source and creates the asymmetry the global params
# must fight. On a hind leg: 30 Nm limps (vx 0.52), 22 Nm makes the incumbent
# fall ~2/3 while a re-tuned low-amplitude gait recovers (vx 0.29, 0 falls),
# 18 Nm the incumbent falls 3/3. 22 Nm is the recoverable, fall-dominated default.
DURATION = 60.0            # continuous bout [s]; damage onset at DURATION/2
DAMAGE_LEG = "RR"          # which leg's hip+knee weaken (a hind leg by default)
HEALTHY_FORCE = 60.0       # hip+knee maxForce before damage [Nm] (~ uncapped)
DAMAGE_FORCE = 22.0        # hip+knee maxForce after damage [Nm]
DAMAGE_RAMP_T = 1.0        # torque droop is ramped over this long [s] (no impulse)
ABD_FORCE = 500.0          # abduction hold force [Nm]; left INTACT (damage models
                           # the sagittal hip+knee actuators, the propulsion ones)

UPRIGHT_FALL = 0.30        # world-up . body-up below which = tip-over
HEIGHT_FALL = 0.25         # base height below which = collapsed [m] (flat ground)


def damage_defaults(duration=DURATION):
    return dict(leg=DAMAGE_LEG, healthy_force=HEALTHY_FORCE,
                damage_force=DAMAGE_FORCE, t_damage=duration / 2.0,
                ramp_t=DAMAGE_RAMP_T)


def load_incumbent():
    with open(FLAT_JSON) as f:
        return np.asarray(json.load(f)["params"], float)


def load_damage_optima():
    """{phase -> 8-vector} pre-fit by fit_damage_oracles.py (oracle arm)."""
    if not os.path.exists(OPTIMA_JSON):
        raise SystemExit("damage_optima.json not found; run fit_damage_oracles.py "
                         "before the oracle arm")
    d = json.load(open(OPTIMA_JSON))
    return {ph: np.asarray(d[ph]["params"], float) for ph in ("healthy", "damaged")}


def leg_force(t, dc):
    """Hip+knee maxForce of the damaged leg at time t: healthy_force until
    t_damage, then ramped down to damage_force over ramp_t (None t_damage =
    never damaged; t_damage<=0 = starts fully damaged -- both used by the fit)."""
    t_dmg = dc["t_damage"]
    hf, df = float(dc["healthy_force"]), float(dc["damage_force"])
    if t_dmg is None or t < t_dmg:
        return hf, 0.0
    frac = min(1.0, (t - t_dmg) / max(dc["ramp_t"], DT))
    return hf + frac * (df - hf), frac


def _fallen_flat(base_pos, base_ori, p):
    rot = p.getMatrixFromQuaternion(base_ori)
    upright = float(np.dot([0, 0, 1], rot[6:]))
    return (upright < UPRIGHT_FALL or base_pos[2] < HEIGHT_FALL), upright


# ── One monitored, squash-adaptive, continuous leg-damage bout ───────────────

# Training-excitation defaults (marxefe arm): a smooth OU dither on the 8 CPG
# parameters while walking safely in phase 1 (the agent's training data). The OU
# correlation time keeps every per-step change small; the dither decays smoothly
# to zero once the trigger fires.
TRAIN_T0 = 4.0             # excitation start [s] (after the 2.4-3.4 s baseline)
TRAIN_STD = 0.05           # stationary dither std, fraction of each param range
TRAIN_TAU = 0.5            # OU correlation time [s]


def run_trial(seed, method_name, k_sigma, incumbent, box, damage_cfg,
              trigger="cusum", duration=DURATION, train=None,
              force_trigger_t=None):
    """One continuous bout on flat ground; at damage_cfg['t_damage'] one leg's
    hip+knee maxForce ramps from healthy_force to damage_force (None = never;
    <=0 = starts damaged -- both used by fit_damage_oracles.py). Trigger / squash
    / safeguard machinery identical to the payload/natural-transect experiments.

    `train` (marxefe arm): dict(t0, std, tau) enabling the phase-1 OU parameter
    excitation; the monitor and the method both see the true (dithered) inputs."""
    import pybullet as p
    from methods import terrain
    from methods.marxefe_optimizer import (get_base_orientation,
                                           load_environment, load_robot)

    terrain.TERRAIN_CONFIG = {"kind": "flat"}
    load_environment(DT, use_gui=False)
    robot, _, joint_IDs_full, _, feet = load_robot(p)

    dc = damage_cfg
    dmg_j = LEG_NAMES.index(dc["leg"])                 # index into range(4)
    dmg_hip = joint_IDs_full[LEG_NAMES[dmg_j]][1]
    dmg_knee = joint_IDs_full[LEG_NAMES[dmg_j]][2]

    # settle (1 s, inside _reset_with_jitter) happens with all legs healthy
    cpg = _reset_with_jitter(p, robot, seed)

    all_joints = [j for leg in LEG_NAMES for j in joint_IDs_full[leg]]
    total_mass = sum(p.getDynamicsInfo(robot, j)[0]
                     for j in range(-1, p.getNumJoints(robot)))

    n_steps = int(round(duration / DT))
    Monitor = {"dt": DecisionTheoreticMonitor, "cusum": CusumDecisionMonitor}.get(
        trigger, TriggerMonitor)
    monitor = Monitor(n_steps, k_sigma)
    is_oracle = (method_name == "oracle")
    optima = load_damage_optima() if is_oracle else None
    method = None if is_oracle else METHODS[method_name](
        np.asarray(incumbent, float), np.asarray(incumbent, float), box, seed)

    t_damage = dc["t_damage"]

    keys = ["t", "x", "y", "z", "vx", "vy", "roll", "pitch", "upright", "power"]
    log = {k: np.zeros(n_steps) for k in keys}
    applied_log = np.zeros((8, n_steps))
    adapting_log = np.zeros(n_steps, dtype=int)
    damage_log = np.zeros(n_steps)

    seg_start = np.asarray(incumbent, float).copy()
    seg_target = seg_start.copy()
    seg_anchor = 0
    applied = seg_start.copy()
    roll = pitch = 0.0
    if is_oracle:                                       # phase-1 optimum from t=0
        seg_target = optima["healthy"].copy()

    trigger_step = None
    guard = None
    window_scores = []
    selected_params = []
    win_buf = {"vx": [], "roll": [], "pitch": []}
    fell, fall_step = False, None
    adapting = True
    n_fires_seen = 0
    n_pauses = 0
    adapt_windows = 0
    propose_times = []
    dmg_frac = 0.0
    dither = np.zeros(8)
    rng_dither = np.random.default_rng(50_000 + int(seed))
    box_lo, box_hi = np.asarray(box[0], float), np.asarray(box[1], float)
    box_rng = box_hi - box_lo

    for k in range(n_steps):
        t = k * DT

        # DAMAGE: hip+knee maxForce of the weak leg, ramped after t_damage.
        legf, dmg_frac = leg_force(t, dc)

        # ORACLE: clairvoyantly switch to the phase-2 optimum at the damage onset.
        if is_oracle and t_damage is not None and t >= t_damage:
            tgt = optima["damaged"]
            if not np.array_equal(tgt, seg_target):
                seg_start = applied.copy()
                seg_target = tgt.copy()
                seg_anchor = k

        # Window boundary (post first trigger): score, propose, squash-pause.
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

        # Training excitation: smooth OU dither on the applied parameters. By
        # default it runs only in phase 1 (pre-trigger) and decays to zero after
        # the trigger. With train["persist"] the excitation CONTINUES after the
        # trigger (optionally at train["persist_std"]) so the agent keeps
        # re-identifying the u->y map under the CHANGED dynamics -- persistent
        # excitation, needed to discover recovery directions in parameters whose
        # effect only appears post-damage (the phase-1 model learns ~zero
        # sensitivity to them). The monitor + method see the true (dithered) u.
        if train is not None:
            a_ou = DT / max(train["tau"], DT)
            exciting = (t >= train["t0"] and
                        (trigger_step is None or train.get("persist")))
            if exciting:
                std = (train["std"] if trigger_step is None
                       else float(train.get("persist_std", train["std"])))
                dither = ((1.0 - a_ou) * dither
                          + np.sqrt(a_ou * (2.0 - a_ou)) * std * box_rng
                          * rng_dither.standard_normal(8))
            else:
                dither *= (1.0 - a_ou)
            applied = np.clip(applied + dither, box_lo, box_hi)

        raw = np.array([int(len(p.getContactPoints(
            bodyA=0, bodyB=robot, linkIndexA=-1, linkIndexB=feet[j])) > 0)
            for j in range(4)])
        hips, knees = cpg.step(applied, raw, DT, roll=roll, pitch=pitch)
        for j in range(4):
            a_id, h_id, k_id = joint_IDs_full[LEG_NAMES[j]]
            p.setJointMotorControl2(robot, a_id, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=ABD_FORCE)
            if j == dmg_j:                              # weakened hip+knee motors
                p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j],
                                        force=legf)
                p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j],
                                        force=legf)
            else:                                       # healthy legs: default force
                p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j])
                p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()

        base_pos, base_ori = get_base_orientation(p, robot, DEFAULT_ORI)
        vel, _ = p.getBaseVelocity(robot)
        pitch, roll, _ = p.getEulerFromQuaternion(base_ori)  # physical (+Y fwd)
        fallen, upright = _fallen_flat(base_pos, base_ori, p)
        js = p.getJointStates(robot, all_joints)
        power = float(sum(abs(s[1] * s[3]) for s in js))     # sum |qdot * tau|

        log["t"][k] = t
        log["x"][k], log["y"][k], log["z"][k] = base_pos[0], base_pos[1], base_pos[2]
        log["vx"][k], log["vy"][k] = vel[1], vel[0]
        log["roll"][k], log["pitch"][k], log["upright"][k] = roll, pitch, upright
        log["power"][k] = power
        applied_log[:, k] = applied
        adapting_log[k] = int(adapting)
        damage_log[k] = dmg_frac

        y_new = np.array([vel[1], vel[0], pitch, roll])
        fired = monitor.step(k, t, y_new, applied)
        if method is not None:
            method.on_step(y_new, applied)

        # `force_trigger_t` (latency diagnostic): start adapting at a chosen time
        # regardless of the monitor -- e.g. exactly at the damage onset to give the
        # online method the oracle's TIMING (but not its params), isolating
        # detection latency from online-search capability.
        force_now = (force_trigger_t is not None and t >= force_trigger_t)
        if not is_oracle and (fired or force_now) and trigger_step is None:
            trigger_step = monitor.fire_step if monitor.fire_step is not None else k
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

    p.disconnect()

    n = n_steps
    for kk in keys:
        log[kk] = log[kk][:n]
    applied_log = applied_log[:, :n]
    adapting_log = adapting_log[:n]
    damage_log = damage_log[:n]

    return dict(
        # per-step signals
        t=log["t"], x=log["x"], y=log["y"], z=log["z"],
        vx=log["vx"], vy=log["vy"], roll=log["roll"], pitch=log["pitch"],
        upright=log["upright"], power=log["power"],
        applied=applied_log, adapting=adapting_log, damage_frac=damage_log,
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
        fell=int(fell),
        adapt_windows=int(adapt_windows), n_pauses=int(n_pauses),
        baseline_mean=float(monitor._baseline_mean()),
        # damage identity
        damage_leg=str(dc["leg"]), healthy_force=float(dc["healthy_force"]),
        damage_force=float(dc["damage_force"]),
        t_shift=(-1.0 if t_damage is None else float(t_damage)),
        damage_ramp_t=float(dc["ramp_t"]), total_mass=float(total_mass),
        # run identity / config
        seed=int(seed), method=method_name, k_sigma=float(k_sigma),
        incumbent=np.asarray(incumbent, float), dt=float(DT),
        window=int(WINDOW), ramp=int(RAMP), duration=float(duration),
        propose_t_total=float(np.sum(propose_times)) if propose_times else 0.0,
    )


# ── Scalar metrics for the manifest ──────────────────────────────────────────

def _tip_dev_deg(roll, pitch):
    return np.rad2deg(np.sqrt(np.asarray(roll) ** 2 + np.asarray(pitch) ** 2))


def scalar_metrics(res):
    n = len(res["y"])
    kT = res["trigger_step"]
    triggered = int(kT >= 0)
    fell = int(res["fell"])
    fall_step = res["fall_step"]
    t = np.asarray(res["t"])
    t_damage = float(res["t_shift"])
    mtail = t >= BASELINE_T[1]                        # after the baseline window
    ph2 = (t >= t_damage) if t_damage >= 0 else np.zeros(n, bool)
    ph1 = mtail & ~ph2
    tip = _tip_dev_deg(res["roll"], res["pitch"])
    vx = np.asarray(res["vx"])
    pw = np.asarray(res["power"])
    g_m = 9.8 * float(res["total_mass"])
    ws = res["window_scores"]
    good = [i for i, J in enumerate(ws) if J >= J_GOOD]
    fall_t = fall_step * DT if fall_step >= 0 else np.nan
    fall_phase = (0 if not fell else (2 if (t_damage >= 0 and fall_t >= t_damage) else 1))
    # detection latency: first monitor fire at/after the damage onset
    fires_t = np.asarray(res["fire_steps"], float) * DT
    post = fires_t[fires_t >= t_damage] if t_damage >= 0 else np.array([])
    det_lat = float(post[0] - t_damage) if len(post) else np.nan

    def _m(x, mask):
        return float(np.mean(np.asarray(x)[mask])) if mask.any() else np.nan

    def _cot(mask):
        v = _m(vx, mask)
        return (_m(pw, mask) / (g_m * v)) if (mask.any() and v and v > 0.05) else np.nan

    return dict(
        seed=res["seed"], method=res["method"],
        triggered=triggered, fell=fell, fall_phase=fall_phase,
        dist=float(res["y"][-1]) if n else 0.0,
        t_end=float(t[-1]) if n else 0.0,
        fall_t=fall_t,
        trigger_t=(kT * DT if triggered else np.nan),
        det_latency=det_lat,
        n_triggers=int(len(res["fire_steps"])),
        n_proposals=int(res["adapt_windows"]),
        n_pauses=int(res["n_pauses"]),
        n_windows=int(len(ws)),
        mean_J=(float(np.mean(ws)) if len(ws) else np.nan),
        win_to_good=(int(good[0]) if good else -1),
        mean_tip_ph1=_m(tip, ph1), mean_tip_ph2=_m(tip, ph2),
        mean_vx_ph1=_m(vx, ph1), mean_vx_ph2=_m(vx, ph2),
        power_ph1=_m(pw, ph1), power_ph2=_m(pw, ph2),
        cot_ph1=_cot(ph1), cot_ph2=_cot(ph2),
        baseline_ce=res["baseline_mean"],
    )


def run_path(seed, method, trigger="ce"):
    suffix = "" if trigger == "ce" else f"_{trigger}"
    return os.path.join(RUNS_DIR, f"dm_seed{seed}_{method}{suffix}.npz")


# ── Job / harness ────────────────────────────────────────────────────────────

def _job(args):
    f2s._limit_threads()
    (seed, method, k_sigma, incumbent, trigger, dt_move, cusum_slack, cusum_h,
     dc, duration, marx_prior_scale, marx_vel_std, marx_forgetting, train) = args
    f2s.DT_BUDGET_MOVE = float(dt_move)
    f2s.MARX_CONTROL_PRIOR_SCALE = float(marx_prior_scale)
    f2s.MARX_GOAL_VEL_STD = float(marx_vel_std)
    f2s.MarxEFE.FORGETTING = float(marx_forgetting)
    if cusum_slack is not None:
        f2s.DT_CUSUM_SLACK = float(cusum_slack)
    if cusum_h is not None:
        f2s.DT_CUSUM_H = float(cusum_h)
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = os.environ.get("CPG_ATTITUDE_FB", "1") != "0"
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    box = (bl.numpy(), bu.numpy())
    res = run_trial(seed, method, k_sigma, incumbent, box, dc,
                    trigger=trigger, duration=duration,
                    train=(train if method == "marxefe" else None))
    path = run_path(seed, method, trigger)
    np.savez_compressed(path, **res)
    row = scalar_metrics(res)
    row["trigger"] = trigger
    row["npz"] = os.path.relpath(path, RESULTS_DIR)
    return row


def run(seeds, arms, k_sigma, workers, trigger="ce", dt_move=DT_BUDGET_MOVE,
        cusum_slack=None, cusum_h=None, leg=DAMAGE_LEG,
        healthy_force=HEALTHY_FORCE, damage_force=DAMAGE_FORCE,
        duration=DURATION,
        marx_prior_scale=MARX_PRIOR_SCALE,
        marx_vel_std=MARX_VEL_STD,
        marx_forgetting=MARX_FORGETTING,
        train_t0=TRAIN_T0, train_std=TRAIN_STD, train_tau=TRAIN_TAU,
        no_train=False):
    os.makedirs(RUNS_DIR, exist_ok=True)
    incumbent = load_incumbent()
    k_eff = 1.0 if trigger in ("dt", "cusum") else float(k_sigma)
    suffix = "" if trigger == "ce" else f"_{trigger}"
    manifest = MANIFEST_CSV if trigger == "ce" else os.path.join(
        RESULTS_DIR, f"manifest{suffix}.csv")
    config = CONFIG_JSON if trigger == "ce" else os.path.join(
        RESULTS_DIR, f"config{suffix}.json")

    dc = damage_defaults(duration)
    dc.update(leg=str(leg), healthy_force=float(healthy_force),
              damage_force=float(damage_force))

    print(f"incumbent (flat-optimal): {np.round(incumbent, 3).tolist()}")
    print(f"trigger={trigger}  seeds={seeds} arms={arms} threshold={k_eff}  "
          f"marx_prior_scale={marx_prior_scale} marx_vel_std={marx_vel_std} "
          f"marx_forgetting={marx_forgetting}")
    train = (None if no_train else
             dict(t0=float(train_t0), std=float(train_std), tau=float(train_tau)))
    if "marxefe" in arms:
        print("marxefe training: " + ("DISABLED (--no-train ablation)" if no_train
              else f"OU std={train['std']}*range, tau={train['tau']} s, "
                   f"from t={train['t0']} s until the trigger fires"))
    print(f"damage: leg={dc['leg']} hip+knee {dc['healthy_force']}->{dc['damage_force']} Nm "
          f"at t={dc['t_damage']} s (ramp {dc['ramp_t']} s); bout {duration} s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(config, "w") as f:
        json.dump(dict(trigger=trigger, seeds=seeds, arms=arms, k_sigma=k_eff,
                       dt_move=dt_move, control_prior_scale=CONTROL_PRIOR_SCALE,
                       marx_prior_scale=marx_prior_scale, marx_vel_std=marx_vel_std,
                       marx_forgetting=marx_forgetting, train=train,
                       damage=dc, dt=DT, window=WINDOW, ramp=RAMP,
                       duration=duration, target_vx=TARGET_VX,
                       incumbent=incumbent.tolist()), f, indent=2)

    jobs = [(int(s), m, k_eff, incumbent, trigger, dt_move, cusum_slack, cusum_h,
             dc, duration, marx_prior_scale, marx_vel_std, marx_forgetting, train)
            for s in range(seeds) for m in arms]
    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=2) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(row)
            tg = "trig" if row["triggered"] else "NO-TRIG"
            end = f"FELL(ph{row['fall_phase']})" if row["fell"] else "cap"
            print(f"[{i+1:3d}/{len(jobs)}] seed{row['seed']:>2} "
                  f"{row['method']:<8} {tg:>7} {end:>9}  "
                  f"dist={row['dist']:.1f}m trig={row['n_triggers']} "
                  f"lat={row['det_latency']:.1f}s "
                  f"vx2={row['mean_vx_ph2']:.2f} P2={row['power_ph2']:.0f}W",
                  flush=True)

    # Merge with an existing manifest: keep rows for (seed, method) pairs that
    # were NOT re-run, so individual arms can be re-run without dropping the rest.
    if os.path.exists(manifest):
        new_keys = {(int(r["seed"]), r["method"]) for r in rows}
        with open(manifest) as f:
            kept = [r for r in csv.DictReader(f)
                    if (int(r["seed"]), r["method"]) not in new_keys]
        if kept:
            print(f"merging {len(kept)} existing rows from {os.path.basename(manifest)}")
            rows.extend(kept)

    rows.sort(key=lambda r: (int(r["seed"]), r["method"]))
    cols = ["seed", "method", "trigger", "triggered", "fell", "fall_phase",
            "dist", "t_end", "trigger_t", "fall_t", "det_latency",
            "n_triggers", "n_proposals", "n_pauses", "n_windows", "mean_J",
            "win_to_good", "mean_tip_ph1", "mean_tip_ph2",
            "mean_vx_ph1", "mean_vx_ph2", "power_ph1", "power_ph2",
            "cot_ph1", "cot_ph2", "baseline_ce", "npz"]
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
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--arms", nargs="+", default=ARMS_DEFAULT, choices=ALL_ARMS)
    ap.add_argument("--K", type=float, default=f2s.K_DEFAULT,
                    help="CE trigger threshold on the baseline-normalised ratio")
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="cusum")
    ap.add_argument("--dt-move", type=float, default=DT_BUDGET_MOVE)
    ap.add_argument("--cusum-slack", type=float, default=None)
    ap.add_argument("--cusum-h", type=float, default=None)
    ap.add_argument("--leg", default=DAMAGE_LEG, choices=["FL", "FR", "RL", "RR"],
                    help="which leg's hip+knee actuators weaken (a HIND leg -- RR "
                         "or RL -- gives a real oracle gap; a front leg is absorbed)")
    ap.add_argument("--healthy-force", type=float, default=HEALTHY_FORCE,
                    help="hip+knee maxForce before damage [Nm] (60 ~ uncapped)")
    ap.add_argument("--damage-force", type=float, default=DAMAGE_FORCE,
                    help="hip+knee maxForce after damage [Nm]. 30~limp, 22~fall-"
                         "but-recoverable (default), <=18 likely unrecoverable")
    ap.add_argument("--duration", type=float, default=DURATION,
                    help="bout length [s]; the leg is damaged at duration/2")
    ap.add_argument("--marx-prior-scale", type=float, default=MARX_PRIOR_SCALE,
                    help="MARX-EFE control-prior width (pull toward incumbent); "
                         "sigma = scale*range/4. 0.5 acts as a trust region")
    ap.add_argument("--marx-vel-std", type=float, default=MARX_VEL_STD,
                    help="MARX-EFE goal-prior std on vx,vy [m/s] (selection only)")
    ap.add_argument("--marx-forgetting", type=float, default=MARX_FORGETTING,
                    help="MARX-EFE model forgetting factor (memory ~1/(1-x) steps)")
    ap.add_argument("--no-train", action="store_true",
                    help="ablation: disable the marxefe phase-1 training excitation")
    ap.add_argument("--train-t0", type=float, default=TRAIN_T0,
                    help="marxefe training: excitation start [s]")
    ap.add_argument("--train-std", type=float, default=TRAIN_STD,
                    help="marxefe training: OU dither stationary std (frac of range)")
    ap.add_argument("--train-tau", type=float, default=TRAIN_TAU,
                    help="marxefe training: OU correlation time [s]")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-attitude-fb", action="store_true",
                    help="disable the CPG's VMC body-attitude feedback (ablation)")
    args = ap.parse_args()
    os.environ["CPG_ATTITUDE_FB"] = "0" if args.no_attitude_fb else "1"
    run(args.seeds, args.arms, args.K, args.workers,
        trigger=args.trigger, dt_move=args.dt_move,
        cusum_slack=args.cusum_slack, cusum_h=args.cusum_h,
        leg=args.leg, healthy_force=args.healthy_force,
        damage_force=args.damage_force, duration=args.duration,
        marx_prior_scale=args.marx_prior_scale,
        marx_vel_std=args.marx_vel_std, marx_forgetting=args.marx_forgetting,
        train_t0=args.train_t0, train_std=args.train_std,
        train_tau=args.train_tau, no_train=args.no_train)


if __name__ == "__main__":
    main()
