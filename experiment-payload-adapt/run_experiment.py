"""Payload-shift online CPG adaptation, triggered by prediction error.

Same triggered / squash-to-stop online-adaptation protocol as
`experiment-flat2slope-adapt`, but the "terrain change" is a change of the
ROBOT'S OWN DYNAMICS, not of the ground: the robot walks on a flat plane
carrying a payload rigidly attached above its trunk. Halfway through the bout
the payload SHIFTS (rearward + lateral, ramped over ~1 s), moving the combined
center of mass off the sagittal plane. Unlike a slope or friction edge, this is
a PERSISTENT mismatch: an asymmetric load costs attitude effort on every stride
until the gait is re-tuned, so suboptimality accumulates instead of being a
one-off transition shock. The transition is also deliberately gentle (ramped
constraint pivot) so the discriminating signal is the sustained mismatch, not
an impulse.

Phases:
  phase 1 (t < t_shift): payload centered over the trunk -- symmetric extra
    mass, the mild condition the incumbent should roughly handle;
  phase 2 (t >= t_shift): payload offset rearward + laterally -- persistent
    roll/pitch torque bias, the condition that should require re-tuning.

Arms (same METHODS objects as flat2slope, single source of truth):
  * noadapt  -> hold the flat-optimal gait throughout (anchor);
  * grid     -> Latin-hypercube search, one window at a time (safeguarded);
  * bo       -> GP-UCB on the windowed stability objective (safeguarded);
  * marxefe  -> active-inference EFE selection, model updated every step;
  * oracle   -> clairvoyant: per-phase pre-fit optimum (fit_payload_oracles.py),
                switched exactly at the shift; upper-bounds any param switch.

The monitor (CE ratio / DT / CUSUM) runs every step; its baseline window
(2.4-3.4 s) lies inside phase 1, so the trigger detects the SHIFT, not the
payload itself. SQUASH pauses adaptation once the error ratio falls back
below K.

Besides falls, per-step actuator mechanical power is logged: with a payload,
"no-adapt limps along inefficiently" is a real outcome that fall counts alone
would miss.

Usage (from repo root):
    python experiment-payload-adapt/run_experiment.py run \
        --seeds 20 --arms noadapt grid bo marxefe --workers 10
    # oracle arm additionally needs: python experiment-payload-adapt/fit_payload_oracles.py
    # then: python experiment-payload-adapt/analyze.py
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
LEG_NAMES = f2s.LEG_NAMES
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
# marxefe_sysid: same MarxEFE method, but with a smooth OU parameter excitation
# during the safe phase 1 so the MARX input coefficients are identified BEFORE
# the shift (otherwise u is constant at the incumbent and the EFE is flat in u).
ALL_ARMS = ARMS_DEFAULT + ["oracle", "marxefe_sysid"]

RESULTS_DIR = os.path.join(_HERE, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
MANIFEST_CSV = os.path.join(RESULTS_DIR, "manifest.csv")
CONFIG_JSON = os.path.join(RESULTS_DIR, "config.json")
OPTIMA_JSON = os.path.join(RESULTS_DIR, "payload_optima.json")

FLAT_JSON = os.path.join(_REPO, "experiment-flat", "results", "selected_params.json")

# ── Payload scenario defaults ────────────────────────────────────────────────
# Screened 2026-07 (see README): at 8 kg, offsets <=0.15 are absorbed by the
# attitude feedback (no trigger), 0.25 is unrecoverable (incumbent falls 3/3;
# only backward-walking gaits survive), 0.20 is the sweet spot: no-adapt limps
# (vx ~0.27 vs an achievable ~0.74) and the CE trigger fires 1-4 s post-shift.
DURATION = 60.0            # continuous bout [s]; shift at DURATION/2
PAYLOAD_MASS = 8.0         # payload mass [kg] (trunk link is 10 kg)
PAYLOAD_UP = 0.15          # payload height above trunk center at attach [m]
SHIFT_LAT = 0.20           # world +X (lateral) shift at t_shift [m]
SHIFT_BACK = 0.20          # world -Y (rearward) shift at t_shift [m]
SHIFT_RAMP_T = 1.0         # shift is ramped over this long [s] (no impulse)
CONSTRAINT_FORCE = 2000.0  # fixed-constraint maxForce [N]

UPRIGHT_FALL = 0.30        # world-up . body-up below which = tip-over
HEIGHT_FALL = 0.25         # base height below which = collapsed [m] (flat ground)


def payload_defaults(duration=DURATION):
    return dict(mass=PAYLOAD_MASS, up=PAYLOAD_UP, lat=SHIFT_LAT, back=SHIFT_BACK,
                t_shift=duration / 2.0, ramp_t=SHIFT_RAMP_T)


def load_incumbent():
    with open(FLAT_JSON) as f:
        return np.asarray(json.load(f)["params"], float)


def load_payload_optima():
    """{phase -> 8-vector} pre-fit by fit_payload_oracles.py (oracle arm)."""
    if not os.path.exists(OPTIMA_JSON):
        raise SystemExit("payload_optima.json not found; run fit_payload_oracles.py "
                         "before the oracle arm")
    d = json.load(open(OPTIMA_JSON))
    return {ph: np.asarray(d[ph]["params"], float) for ph in ("centered", "shifted")}


def attach_payload(p, robot, mass, up=PAYLOAD_UP, lat=0.0, back=0.0, frac0=0.0):
    """Rigidly attach a payload box `up` meters above the trunk center, already
    displaced by frac0 * (lat, -back) in the horizontal plane (frac0=1 is the
    fully-shifted condition used for oracle fitting -- the box is CREATED at the
    shifted spot so the constraint is exactly satisfied from step 0, no yank).

    The box has a proper box inertia but is collision-masked off (pure mass:
    it can never snag on the robot or the ground). Returns (body_id, cid).
    The payload's own frame is world-aligned at creation and the JOINT_FIXED
    constraint locks the relative orientation, so a child-pivot of -s moves the
    payload by +s in (approximately) world/body-horizontal coordinates -- see
    shift_pivot()."""
    half = 0.06
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
    base_pos, base_quat = p.getBasePositionAndOrientation(robot)
    anchor_off = np.array([0.0, 0.0, float(up)])       # centered anchor (trunk frame)
    child_pivot = shift_pivot(lat, back, frac0)
    payload = p.createMultiBody(baseMass=float(mass),
                                baseCollisionShapeIndex=col,
                                basePosition=(np.asarray(base_pos) + anchor_off
                                              - np.asarray(child_pivot)).tolist())
    p.setCollisionFilterGroupMask(payload, -1, 0, 0)   # mass only, no contacts
    rot = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
    parent_pivot = rot.T @ anchor_off                  # trunk-local attach point
    cid = p.createConstraint(robot, -1, payload, -1, p.JOINT_FIXED,
                             [0, 0, 0], parent_pivot.tolist(), child_pivot)
    p.changeConstraint(cid, maxForce=CONSTRAINT_FORCE)
    return payload, cid


def shift_pivot(lat, back, frac):
    """Child-frame pivot realising a payload displacement of frac * (lat, -back)
    in the horizontal plane (robot walks +Y, so `back` moves the load rearward).
    payload_pos = anchor - R_payload . child_pivot with R_payload ~ locked to the
    (level) trunk, so child_pivot = -shift."""
    return [-frac * float(lat), frac * float(back), 0.0]


def _fallen_flat(base_pos, base_ori, p):
    rot = p.getMatrixFromQuaternion(base_ori)
    upright = float(np.dot([0, 0, 1], rot[6:]))
    return (upright < UPRIGHT_FALL or base_pos[2] < HEIGHT_FALL), upright


# ── One monitored, squash-adaptive, continuous payload bout ──────────────────

# System-identification excitation defaults (marxefe_sysid arm): a smooth OU
# dither on the 8 CPG parameters while walking safely in phase 1, so the MARX
# model sees VARIED inputs and identifies the u -> y map before the shift. The
# OU correlation time keeps every per-step change small (slow change, no jumps);
# the dither decays smoothly to zero once the trigger fires.
SYSID_T0 = 4.0             # excitation start [s] (after the 2.4-3.4 s baseline)
SYSID_STD = 0.05           # stationary dither std, fraction of each param range
SYSID_TAU = 0.5            # OU correlation time [s]


def run_trial(seed, method_name, k_sigma, incumbent, box, payload_cfg,
              trigger="ce", duration=DURATION, sysid=None):
    """One continuous bout on flat ground with an attached payload that shifts
    at payload_cfg['t_shift'] (None = never shifts; 0.0 = starts shifted --
    both used by fit_payload_oracles.py). Trigger / squash / safeguard machinery
    identical to the natural-transect experiment. Returns per-step signals.

    `sysid` (marxefe_sysid arm): dict(t0, std, tau) enabling the phase-1 OU
    parameter excitation described above; the monitor and the method both see
    the true (dithered) inputs."""
    import pybullet as p
    from methods import terrain
    from methods.marxefe_optimizer import (get_base_orientation,
                                           load_environment, load_robot)

    terrain.TERRAIN_CONFIG = {"kind": "flat"}
    load_environment(DT, use_gui=False)
    robot, _, joint_IDs_full, _, feet = load_robot(p)

    pc = payload_cfg
    t_shift = pc["t_shift"]
    starts_shifted = (t_shift is not None and t_shift <= 0.0)  # fitting condition
    _, cid = attach_payload(p, robot, pc["mass"], up=pc["up"],
                            lat=pc["lat"], back=pc["back"],
                            frac0=(1.0 if starts_shifted else 0.0))
    if starts_shifted:
        t_shift = None
    # settle (1 s, inside _reset_with_jitter) happens WITH the payload attached
    cpg = _reset_with_jitter(p, robot, seed)

    all_joints = [j for leg in LEG_NAMES for j in joint_IDs_full[leg]]
    total_mass = sum(p.getDynamicsInfo(robot, j)[0]
                     for j in range(-1, p.getNumJoints(robot))) + pc["mass"]

    n_steps = int(round(duration / DT))
    Monitor = {"dt": DecisionTheoreticMonitor, "cusum": CusumDecisionMonitor}.get(
        trigger, TriggerMonitor)
    monitor = Monitor(n_steps, k_sigma)
    is_oracle = (method_name == "oracle")
    base_name = method_name[:-6] if method_name.endswith("_sysid") else method_name
    optima = load_payload_optima() if is_oracle else None
    method = None if is_oracle else METHODS[base_name](
        np.asarray(incumbent, float), np.asarray(incumbent, float), box, seed)

    keys = ["t", "x", "y", "z", "vx", "vy", "roll", "pitch", "upright", "power"]
    log = {k: np.zeros(n_steps) for k in keys}
    applied_log = np.zeros((8, n_steps))
    adapting_log = np.zeros(n_steps, dtype=int)
    shift_log = np.zeros(n_steps)

    seg_start = np.asarray(incumbent, float).copy()
    seg_target = seg_start.copy()
    seg_anchor = 0
    applied = seg_start.copy()
    roll = pitch = 0.0
    if is_oracle:                                       # phase-1 optimum from t=0
        seg_target = optima["centered"].copy()

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
    shift_frac = 1.0 if starts_shifted else 0.0
    dither = np.zeros(8)
    rng_dither = np.random.default_rng(50_000 + int(seed))
    box_lo, box_hi = np.asarray(box[0], float), np.asarray(box[1], float)
    box_rng = box_hi - box_lo

    for k in range(n_steps):
        t = k * DT

        # PAYLOAD SHIFT: ramp the constraint pivot over ramp_t seconds.
        if t_shift is not None and t >= t_shift and shift_frac < 1.0:
            shift_frac = min(1.0, (t - t_shift) / max(pc["ramp_t"], DT))
            p.changeConstraint(cid,
                               jointChildPivot=shift_pivot(pc["lat"], pc["back"],
                                                           shift_frac),
                               maxForce=CONSTRAINT_FORCE)

        # ORACLE: clairvoyantly switch to the phase-2 optimum at the shift.
        if is_oracle and t_shift is not None and t >= t_shift:
            tgt = optima["shifted"]
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

        # SYSID excitation: smooth OU dither on the applied parameters while in
        # phase 1 (pre-trigger); decays smoothly to zero after the trigger so no
        # parameter ever changes instantaneously. Monitor + method see the true u.
        if sysid is not None:
            a_ou = DT / max(sysid["tau"], DT)
            if trigger_step is None and t >= sysid["t0"]:
                dither = ((1.0 - a_ou) * dither
                          + np.sqrt(a_ou * (2.0 - a_ou)) * sysid["std"] * box_rng
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
                                    targetPosition=0.0, force=500)
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
        shift_log[k] = shift_frac

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

    p.disconnect()

    n = n_steps
    for kk in keys:
        log[kk] = log[kk][:n]
    applied_log = applied_log[:, :n]
    adapting_log = adapting_log[:n]
    shift_log = shift_log[:n]

    return dict(
        # per-step signals
        t=log["t"], x=log["x"], y=log["y"], z=log["z"],
        vx=log["vx"], vy=log["vy"], roll=log["roll"], pitch=log["pitch"],
        upright=log["upright"], power=log["power"],
        applied=applied_log, adapting=adapting_log, shift_frac=shift_log,
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
        # payload identity
        payload_mass=float(pc["mass"]), shift_lat=float(pc["lat"]),
        shift_back=float(pc["back"]),
        t_shift=(-1.0 if t_shift is None else float(t_shift)),
        shift_ramp_t=float(pc["ramp_t"]), total_mass=float(total_mass),
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
    t_shift = float(res["t_shift"])
    mtail = t >= BASELINE_T[1]                        # after the baseline window
    ph2 = (t >= t_shift) if t_shift >= 0 else np.zeros(n, bool)
    ph1 = mtail & ~ph2
    tip = _tip_dev_deg(res["roll"], res["pitch"])
    vx = np.asarray(res["vx"])
    pw = np.asarray(res["power"])
    g_m = 9.8 * float(res["total_mass"])
    ws = res["window_scores"]
    good = [i for i, J in enumerate(ws) if J >= J_GOOD]
    fall_t = fall_step * DT if fall_step >= 0 else np.nan
    fall_phase = (0 if not fell else (2 if (t_shift >= 0 and fall_t >= t_shift) else 1))
    # detection latency: first monitor fire at/after the shift
    fires_t = np.asarray(res["fire_steps"], float) * DT
    post = fires_t[fires_t >= t_shift] if t_shift >= 0 else np.array([])
    det_lat = float(post[0] - t_shift) if len(post) else np.nan

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
    return os.path.join(RUNS_DIR, f"pl_seed{seed}_{method}{suffix}.npz")


# ── Job / harness ────────────────────────────────────────────────────────────

def _job(args):
    f2s._limit_threads()
    (seed, method, k_sigma, incumbent, trigger, dt_move, cusum_slack, cusum_h,
     pc, duration, marx_prior_scale, marx_vel_std, marx_forgetting, sysid) = args
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
    res = run_trial(seed, method, k_sigma, incumbent, box, pc,
                    trigger=trigger, duration=duration,
                    sysid=(sysid if method.endswith("_sysid") else None))
    path = run_path(seed, method, trigger)
    np.savez_compressed(path, **res)
    row = scalar_metrics(res)
    row["trigger"] = trigger
    row["npz"] = os.path.relpath(path, RESULTS_DIR)
    return row


def run(seeds, arms, k_sigma, workers, trigger="ce", dt_move=DT_BUDGET_MOVE,
        cusum_slack=None, cusum_h=None, mass=PAYLOAD_MASS, lat=SHIFT_LAT,
        back=SHIFT_BACK, duration=DURATION,
        marx_prior_scale=f2s.MARX_CONTROL_PRIOR_SCALE,
        marx_vel_std=f2s.MARX_GOAL_VEL_STD,
        marx_forgetting=f2s.MarxEFE.FORGETTING,
        sysid_t0=SYSID_T0, sysid_std=SYSID_STD, sysid_tau=SYSID_TAU):
    os.makedirs(RUNS_DIR, exist_ok=True)
    incumbent = load_incumbent()
    k_eff = 1.0 if trigger in ("dt", "cusum") else float(k_sigma)
    suffix = "" if trigger == "ce" else f"_{trigger}"
    manifest = MANIFEST_CSV if trigger == "ce" else os.path.join(
        RESULTS_DIR, f"manifest{suffix}.csv")
    config = CONFIG_JSON if trigger == "ce" else os.path.join(
        RESULTS_DIR, f"config{suffix}.json")

    pc = payload_defaults(duration)
    pc.update(mass=float(mass), lat=float(lat), back=float(back))

    print(f"incumbent (flat-optimal): {np.round(incumbent, 3).tolist()}")
    print(f"trigger={trigger}  seeds={seeds} arms={arms} threshold={k_eff}  "
          f"marx_prior_scale={marx_prior_scale} marx_vel_std={marx_vel_std} "
          f"marx_forgetting={marx_forgetting}")
    sysid = dict(t0=float(sysid_t0), std=float(sysid_std), tau=float(sysid_tau))
    if any(a.endswith("_sysid") for a in arms):
        print(f"sysid excitation: OU std={sysid['std']}*range, tau={sysid['tau']} s, "
              f"from t={sysid['t0']} s until the trigger fires")
    print(f"payload: mass={pc['mass']} kg, shift=({pc['lat']} lat, {pc['back']} back) m "
          f"at t={pc['t_shift']} s (ramp {pc['ramp_t']} s); bout {duration} s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(config, "w") as f:
        json.dump(dict(trigger=trigger, seeds=seeds, arms=arms, k_sigma=k_eff,
                       dt_move=dt_move, control_prior_scale=CONTROL_PRIOR_SCALE,
                       marx_prior_scale=marx_prior_scale, marx_vel_std=marx_vel_std,
                       marx_forgetting=marx_forgetting, sysid=sysid,
                       payload=pc, dt=DT, window=WINDOW, ramp=RAMP,
                       duration=duration, target_vx=TARGET_VX,
                       incumbent=incumbent.tolist()), f, indent=2)

    jobs = [(int(s), m, k_eff, incumbent, trigger, dt_move, cusum_slack, cusum_h,
             pc, duration, marx_prior_scale, marx_vel_std, marx_forgetting, sysid)
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
    ap.add_argument("--trigger", choices=["ce", "dt", "cusum"], default="ce")
    ap.add_argument("--dt-move", type=float, default=DT_BUDGET_MOVE)
    ap.add_argument("--cusum-slack", type=float, default=None)
    ap.add_argument("--cusum-h", type=float, default=None)
    ap.add_argument("--mass", type=float, default=PAYLOAD_MASS,
                    help="payload mass [kg]")
    ap.add_argument("--shift-lat", type=float, default=SHIFT_LAT,
                    help="lateral payload shift at t_shift [m]")
    ap.add_argument("--shift-back", type=float, default=SHIFT_BACK,
                    help="rearward payload shift at t_shift [m]")
    ap.add_argument("--duration", type=float, default=DURATION,
                    help="bout length [s]; the payload shifts at duration/2")
    ap.add_argument("--marx-prior-scale", type=float,
                    default=f2s.MARX_CONTROL_PRIOR_SCALE,
                    help="MARX-EFE control-prior width (pull toward incumbent); "
                         "sigma = scale*range/4. Default 0.15 pins proposals to "
                         "the incumbent; ~1-3 allows optimum-sized moves")
    ap.add_argument("--marx-vel-std", type=float, default=f2s.MARX_GOAL_VEL_STD,
                    help="MARX-EFE goal-prior std on vx,vy [m/s] (selection only; "
                         "the trigger monitors keep MON_VEL_STD). Default 1.0 makes "
                         "a ~0.15 m/s limp invisible to the EFE; ~0.2 makes the "
                         "velocity deficit drive adaptation")
    ap.add_argument("--marx-forgetting", type=float,
                    default=f2s.MarxEFE.FORGETTING,
                    help="MARX-EFE model forgetting factor (memory ~1/(1-x) steps). "
                         "Default 0.995 = 2 s; the sysid arm wants ~0.999 = 10 s so "
                         "the phase-1 excitation data survives until adaptation")
    ap.add_argument("--sysid-t0", type=float, default=SYSID_T0,
                    help="marxefe_sysid: excitation start [s] (keep after the "
                         "3.4 s trigger baseline)")
    ap.add_argument("--sysid-std", type=float, default=SYSID_STD,
                    help="marxefe_sysid: OU dither stationary std, fraction of "
                         "each parameter's range")
    ap.add_argument("--sysid-tau", type=float, default=SYSID_TAU,
                    help="marxefe_sysid: OU correlation time [s] (slow change)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-attitude-fb", action="store_true",
                    help="disable the CPG's VMC body-attitude feedback (ablation)")
    args = ap.parse_args()
    os.environ["CPG_ATTITUDE_FB"] = "0" if args.no_attitude_fb else "1"
    run(args.seeds, args.arms, args.K, args.workers,
        trigger=args.trigger, dt_move=args.dt_move,
        cusum_slack=args.cusum_slack, cusum_h=args.cusum_h,
        mass=args.mass, lat=args.shift_lat, back=args.shift_back,
        duration=args.duration, marx_prior_scale=args.marx_prior_scale,
        marx_vel_std=args.marx_vel_std, marx_forgetting=args.marx_forgetting,
        sysid_t0=args.sysid_t0, sysid_std=args.sysid_std,
        sysid_tau=args.sysid_tau)


if __name__ == "__main__":
    main()
