"""Continual (non-episodic) payload-shift recovery with the GP-safe agent.

The continual analogue of `experiment-damage-adapt/run_continual.py`, with the
recurring perturbation being a payload CoM SHIFT instead of a leg-torque droop.

One long continuous bout. The robot walks carrying an 8 kg trunk payload; at
random intervals the payload SHIFTS off the sagittal plane (rearward + lateral,
ramped ~1 s). The moment the shift sets in, the agent reacts by proposing a CPG
gait from its GP memory over the control space (safe acquisition that avoids the
parameter regions it remembers falling). Each shift event is one query, folded
straight back into the memory:

  * if the robot FALLS   -> record (gait, V_FALL); stand it back upright and
    RECENTER the payload; after a random 2-8 s of walking the shift recurs;
  * if it SURVIVES a hold window under the full shift -> record (gait, V);
    recenter, and the shift recurs after the same random delay.

So the memory persists across recurrences WITHIN the single trial: the agent
should stop re-trying gaits it has already seen fall and converge onto a
recovery gait. No gait is ever supplied. Detection is a prediction-error CUSUM
(forward-speed deficit + extra tilt vs the pre-shift baseline), re-armed after
each recenter -- the same trigger family used across this repo.

The payload attach/shift/fall-check helpers are imported unchanged from
`run_experiment.py`; the GP-safe agent is `methods.gp_safe_agent`.

Usage (from repo root):
    python experiment-payload-adapt/run_continual.py --duration 120 --seeds 3
"""

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
PARAM_NAMES = ["coupling", "w_swing", "w_stance", "F_FAST", "STOP", "hipA", "kneeA", "b"]

# The CPG dims the healthy<->shifted optimum actually moves (from the oracle
# refit: coupling, w_swing, F_FAST, STOP, b differ; w_stance/hip/knee barely).
FREE_DIMS_PAYLOAD = [0, 1, 3, 4, 7]

# event / scheduling constants
PARAM_RAMP = 30          # steps to ramp a gait switch in (0.3 s)
EVAL_HOLD = 3.0          # survive this long under full shift -> success [s]
GAP_MIN, GAP_MAX = 2.0, 8.0   # random interval before the shift recurs [s]
FIRST_SHIFT_T = 4.0      # first shift onset [s]
SETTLE_STEPS = 60        # upright-reset settle [steps]
ARM_TIMEOUT = 6.0        # force-arm the detector this long after a heal even if
                         # the health gate never clears (liveness backstop). Long
                         # enough that the health gate normally arms first once the
                         # robot recovers -- a shorter value arms mid-transient and
                         # false-alarms on the recovery deficit.
DETECT_TIMEOUT = 3.0     # force a response if an engaged shift goes undetected
                         # this long without a fall (liveness: no stuck limp)
TARGET_VX = 0.5
V_FALL = -2.0


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _score(vx, roll, pitch):
    """Post-shift V over an event window (same scale as the GP memory)."""
    if len(vx) < 20:
        return V_FALL
    r_v = min(max(float(np.mean(vx)), 0.0) / TARGET_VX, 1.0)
    rms_roll = np.rad2deg(np.sqrt(np.mean(np.asarray(roll) ** 2)))
    rms_pitch = np.rad2deg(np.sqrt(np.mean(np.asarray(pitch) ** 2)))
    return float(r_v - (rms_roll + rms_pitch) / 10.0)


def run_seed(seed, duration, pl, gp, free_dims, n_init, args):
    import pybullet as p
    from methods import terrain
    from methods.marxefe_optimizer import (get_base_orientation,
                                           load_environment, load_robot, JointCPG)
    DT = pl.DT
    LEG = pl.LEG_NAMES
    ORI = pl.DEFAULT_ORI
    UP = pl.PAYLOAD_UP
    incumbent = pl.load_incumbent()
    from methods.cpg_bounds import bounds_lower, bounds_upper
    box = (bounds_lower.numpy(), bounds_upper.numpy())
    mass, lat, back = args.mass, args.lat, args.back

    JointCPG.ATTITUDE_FEEDBACK = True
    terrain.TERRAIN_CONFIG = {"kind": "flat"}
    load_environment(DT, use_gui=False)
    robot, _, jids, _, feet = load_robot(p)
    payload, cid = pl.attach_payload(p, robot, mass, up=UP, lat=lat, back=back,
                                     frac0=0.0)

    def recenter_payload(at_xy):
        """Put the payload back at the centered anchor above the trunk and clear
        the shift (child pivot -> 0), so a heal/reset introduces no yank."""
        p.changeConstraint(cid, jointChildPivot=pl.shift_pivot(lat, back, 0.0),
                           maxForce=pl.CONSTRAINT_FORCE)
        p.resetBasePositionAndOrientation(
            payload, [at_xy[0], at_xy[1], 0.55 + UP], [0, 0, 0, 1])
        p.resetBaseVelocity(payload, [0, 0, 0], [0, 0, 0])

    def settle(at_xy, s):
        """Stand the robot upright at (x,y) and recenter the payload; fresh CPG."""
        rng = np.random.default_rng(10_000 + int(s))
        jit = rng.normal(0.0, 0.002, size=12)
        p.resetBasePositionAndOrientation(robot, [at_xy[0], at_xy[1], 0.55], ORI)
        p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])
        recenter_payload(at_xy)
        abd, hip, kn = [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14]
        for i, j in enumerate(abd):
            p.resetJointState(robot, j, 0.0 + jit[i])
        for i, j in enumerate(hip):
            p.resetJointState(robot, j, 0.05 + jit[4 + i])
        for i, j in enumerate(kn):
            p.resetJointState(robot, j, -0.6 + jit[8 + i])
        for _ in range(SETTLE_STEPS):
            for j in abd:
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=0.0, force=500)
            for j in hip:
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, 0.25)
            for j in kn:
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, -1.0)
            p.stepSimulation()
        return JointCPG(n_legs=4)

    cpg = settle([0.0, 0.0], seed)
    agent = gp.GPSafeRecovery(incumbent, box, free_dims=free_dims, seed=seed,
                              n_init=n_init, safe_V=args.safe_V, beta=args.beta,
                              kappa=args.kappa, objective=args.objective,
                              efe_y_star=args.efe_y_star, efe_tau2=args.efe_tau2,
                              efe_adaptive=args.efe_adaptive,
                              efe_tau2_min=args.efe_tau2_min,
                              efe_tau2_max=args.efe_tau2_max, archive_path=None)
    agent.update(incumbent, V_FALL, True)        # seed: incumbent is unsafe under shift
    rng = np.random.default_rng(777 + seed)

    n_steps = int(round(duration / DT))
    log = {k: np.zeros(n_steps) for k in
           ("t", "y", "vx", "roll", "pitch", "shift", "state", "cusum")}
    events = []

    # ── prediction-error detector (CUSUM on a smoothed health signal) ────────
    use_det = not args.no_detector
    a_s = DT / max(args.detect_tau, DT)          # EMA smoothing coefficient
    TIP_SCALE = np.deg2rad(12.0)
    vx_s = tip_s = None
    S = 0.0
    vx_base = tip_base = None
    warm_vx, warm_tip = [], []

    phase = "monitoring"                          # or "responding"
    shift_active = False                          # physical shift engaged
    shift_frac = 0.0                              # 0 centered .. 1 fully shifted
    onset_t = None
    detect_t = None
    armed = False
    heal_streak = 0
    ARM_STREAK = int(round(0.5 / DT))
    next_shift_t = FIRST_SHIFT_T
    grace_until = 1.5
    last_event_t = 0.0                            # for the force-arm watchdog
    cand = incumbent.copy()
    seg_start = incumbent.copy(); seg_target = incumbent.copy(); seg_anchor = 0
    applied = incumbent.copy()
    win = {"vx": [], "roll": [], "pitch": []}
    roll = pitch = 0.0
    n_false = 0
    n_reset = 0

    def _resolve(fell, V, base_pos, k, t):
        """Record the outcome, recenter the payload, (reset upright if fell), disarm."""
        nonlocal phase, shift_active, next_shift_t, grace_until, cpg
        nonlocal seg_start, seg_target, seg_anchor, win, roll, pitch, S
        nonlocal onset_t, detect_t, armed, last_event_t
        # Only real shift responses (onset set) update the GP recovery map; a
        # false alarm evaluates a gait under the CENTERED load, whose V lives on a
        # different surface -- fold it in and it poisons the memory. Still logged.
        if onset_t is not None:
            agent.update(cand, V, fell)
        events.append(dict(onset=onset_t, detect=detect_t,
                           latency=(detect_t - onset_t if (detect_t is not None
                                    and onset_t is not None) else np.nan),
                           fell=int(fell), V=float(V), cand=cand.copy(),
                           false_alarm=(onset_t is None)))
        if fell:
            cpg = settle([base_pos[0], base_pos[1]], seed * 131 + len(events))
            roll = pitch = 0.0
            seg_start = incumbent.copy()
        else:
            seg_start = None                       # filled below from `applied`
        seg_target = incumbent.copy()
        seg_anchor = k + 1
        win = {"vx": [], "roll": [], "pitch": []}
        shift_active = False                        # heal: payload ramps back to centered
        onset_t = None; detect_t = None
        armed = False
        phase = "monitoring"
        next_shift_t = t + rng.uniform(GAP_MIN, GAP_MAX)
        grace_until = t + args.grace
        last_event_t = t
        S = 0.0

    k = 0
    while k < n_steps:
        t = k * DT
        # physical shift schedule (agent is blind to this). Gated on the detector
        # being armed -- i.e. the robot has returned to confirmed-healthy walking
        # after the last heal -- so every shift lands over a live monitor (no
        # undetected-limp stuck state) and recurs "in the next 2-8 s of walking".
        if (phase == "monitoring" and not shift_active and armed
                and t >= next_shift_t):
            shift_active = True; onset_t = t; detect_t = None
        # ramp the shift toward its target (1 engaged, 0 centered), no jump
        target_frac = 1.0 if shift_active else 0.0
        step_frac = DT / max(pl.SHIFT_RAMP_T, DT)
        shift_frac = float(np.clip(shift_frac + np.sign(target_frac - shift_frac)
                                   * step_frac, 0.0, 1.0))
        p.changeConstraint(cid, jointChildPivot=pl.shift_pivot(lat, back, shift_frac),
                           maxForce=pl.CONSTRAINT_FORCE)

        frac_p = min(1.0, (k - seg_anchor) / max(1, PARAM_RAMP))
        if seg_start is None:                      # survived-heal: ramp from current
            seg_start = applied.copy()
        applied = seg_start + frac_p * (seg_target - seg_start)

        raw = np.array([int(len(p.getContactPoints(bodyA=0, bodyB=robot,
              linkIndexA=-1, linkIndexB=feet[j])) > 0) for j in range(4)])
        hips, knees = cpg.step(applied, raw, DT, roll=roll, pitch=pitch)
        for j in range(4):
            a_id, h_id, k_id = jids[LEG[j]]
            p.setJointMotorControl2(robot, a_id, p.POSITION_CONTROL, targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j])
            p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()

        base_pos, base_ori = get_base_orientation(p, robot, ORI)
        vel, _ = p.getBaseVelocity(robot)
        pitch, roll, _ = p.getEulerFromQuaternion(base_ori)
        fell, _up = pl._fallen_flat(base_pos, base_ori, p)
        vx = vel[1]; tipmag = np.hypot(roll, pitch)

        vx_s = vx if vx_s is None else vx_s + a_s * (vx - vx_s)
        tip_s = tipmag if tip_s is None else tip_s + a_s * (tipmag - tip_s)
        if vx_base is None:
            if 1.5 <= t < FIRST_SHIFT_T:
                warm_vx.append(vx); warm_tip.append(tipmag)
            if t >= FIRST_SHIFT_T - DT:
                vx_base = max(float(np.mean(warm_vx)), 0.1) if warm_vx else 0.5
                tip_base = float(np.mean(warm_tip)) if warm_tip else 0.0

        # Arm once re-stabilized after a heal. Threshold 0.70 (not 0.85) of the
        # healthy baseline: per-stride ripple in the EMA speed otherwise keeps
        # breaking the 0.5 s streak and the detector never re-arms. The shift
        # deficit is far larger than this margin, so detection is unaffected.
        healthy_now = (vx_base is not None and phase == "monitoring"
                       and t >= grace_until and vx_s >= 0.70 * vx_base
                       and tip_s <= tip_base + np.deg2rad(5.0))
        heal_streak = heal_streak + 1 if healthy_now else 0
        # arm on a sustained healthy streak, OR force-arm after a bounded wait so
        # a robot that never fully recovers to baseline can't stall the schedule
        if not armed and (heal_streak >= ARM_STREAK
                          or (vx_base is not None and phase == "monitoring"
                              and t >= last_event_t + ARM_TIMEOUT)):
            armed = True; S = 0.0
        e = 0.0
        if armed and phase == "monitoring":
            e = (max(0.0, vx_base - vx_s) / vx_base
                 + max(0.0, tip_s - tip_base) / TIP_SCALE)
            S = max(0.0, S + e - args.detect_kappa)
        log["t"][k], log["y"][k] = t, base_pos[1]
        log["vx"][k], log["roll"][k], log["pitch"][k] = vx, roll, pitch
        log["shift"][k] = shift_frac
        log["state"][k] = 1.0 if phase == "responding" else 0.0
        log["cusum"][k] = S

        if phase == "responding":
            if t - detect_t > PARAM_RAMP * DT + 0.2:
                win["vx"].append(vx); win["roll"].append(roll); win["pitch"].append(pitch)
            if fell:
                _resolve(True, V_FALL, base_pos, k, t)
            elif t - detect_t >= PARAM_RAMP * DT + EVAL_HOLD:
                _resolve(False, _score(win["vx"], win["roll"], win["pitch"]),
                         base_pos, k, t)
        else:  # monitoring
            fired = (armed and S > args.detect_h) if use_det \
                else (shift_active and detect_t is None)
            # liveness: if a shift is engaged but the detector hasn't fired within
            # DETECT_TIMEOUT (a missed/limp shift), force a (late) response so the
            # agent still acts and the schedule cannot stall in a shifted state.
            if (use_det and shift_active and detect_t is None and not fired
                    and onset_t is not None and t - onset_t > DETECT_TIMEOUT):
                fired = True
            if fell and onset_t is not None:             # undetected shift fall
                cand = incumbent.copy()
                _resolve(True, V_FALL, base_pos, k, t)
            elif fell:                                   # residual healthy fall
                cpg = settle([base_pos[0], base_pos[1]], seed * 131 + 90000 + k)
                roll = pitch = 0.0
                seg_start = incumbent.copy(); seg_target = incumbent.copy()
                seg_anchor = k + 1
                armed = False; heal_streak = 0; S = 0.0
                grace_until = t + args.grace
                n_reset += 1
            elif fired:
                detect_t = t
                if onset_t is None:
                    n_false += 1
                cand, _mode = agent.propose()
                seg_start = applied.copy(); seg_target = cand.copy(); seg_anchor = k + 1
                win = {"vx": [], "roll": [], "pitch": []}
                phase = "responding"; armed = False; S = 0.0
        k += 1

    p.disconnect()
    for key in log:
        log[key] = log[key][:k]
    return log, events, agent, n_false, n_reset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mass", type=float, default=8.0, help="payload mass [kg]")
    ap.add_argument("--lat", type=float, default=0.215, help="lateral shift [m]")
    ap.add_argument("--back", type=float, default=0.20, help="rearward shift [m]")
    ap.add_argument("--free-dims", type=int, nargs="+", default=None)
    ap.add_argument("--n-init", type=int, default=4)
    ap.add_argument("--safe-V", type=float, default=-0.8)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--objective", choices=["ucb", "efe"], default="efe",
                    help="agent planning objective: GP-UCB or Expected Free Energy")
    ap.add_argument("--efe-y-star", type=float, default=1.0)
    ap.add_argument("--efe-tau2", type=float, default=0.5)
    ap.add_argument("--efe-adaptive", action="store_true")
    ap.add_argument("--efe-tau2-min", type=float, default=0.1)
    ap.add_argument("--efe-tau2-max", type=float, default=3.0)
    ap.add_argument("--no-detector", action="store_true",
                    help="idealised: react at shift onset instead of detecting")
    ap.add_argument("--detect-kappa", type=float, default=0.20,
                    help="CUSUM slack (tolerated per-step prediction error); "
                         "must exceed the recovery-transient deficit to avoid "
                         "post-event false alarms")
    ap.add_argument("--detect-h", type=float, default=1.8,
                    help="CUSUM decision threshold (fire when accumulator > h)")
    ap.add_argument("--detect-tau", type=float, default=0.4,
                    help="smoothing time constant for the health signal [s]")
    ap.add_argument("--grace", type=float, default=2.5,
                    help="post-recenter grace before re-arming the detector [s]")
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pl = _load("run_experiment")
    from methods import gp_safe_agent as gp
    free = a.free_dims if a.free_dims is not None else FREE_DIMS_PAYLOAD
    print(f"continual payload-shift recovery: {a.seeds} seeds x {a.duration:g}s, "
          f"{a.mass:g}kg shift ({a.lat:g} lat, {a.back:g} back) recurring every "
          f"{GAP_MIN:g}-{GAP_MAX:g}s")
    obj = (f"EFE (y*={a.efe_y_star}, tau2="
           f"{'adaptive[%.1f,%.1f]' % (a.efe_tau2_min, a.efe_tau2_max) if a.efe_adaptive else a.efe_tau2})"
           if a.objective == "efe" else f"GP-UCB (beta={a.beta})")
    det = ("idealised (react at onset)" if a.no_detector else
           f"prediction-error CUSUM (kappa={a.detect_kappa}, h={a.detect_h}, "
           f"tau={a.detect_tau}s)")
    print(f"  objective: {obj}")
    print(f"  detector: {det}")
    print(f"  searching dims {free} ({[PARAM_NAMES[i] for i in free]}); "
          f"upright-reset + recenter on fall; hold {EVAL_HOLD:g}s to confirm survival\n")

    all_events = {}
    logs = {}
    n_false_tot = 0
    for s in range(a.seeds):
        log, events, agent, n_false, n_reset = run_seed(s, a.duration, pl, gp, free, a.n_init, a)
        all_events[s] = events
        logs[s] = log
        n_false_tot += n_false
        n = len(events)
        nf = sum(e["fell"] for e in events)
        third = max(1, n // 3)
        fr_early = np.mean([e["fell"] for e in events[:third]])
        fr_late = np.mean([e["fell"] for e in events[-third:]])
        bestV = max([e["V"] for e in events], default=float("nan"))
        lats = [e["latency"] for e in events if e["latency"] == e["latency"]]
        mlat = float(np.mean(lats)) if lats else float("nan")
        print(f"seed {s}: {n} shift events, {nf} falls ({nf/max(n,1):.0%}); "
              f"fall rate first third {fr_early:.0%} -> last third {fr_late:.0%}; "
              f"best V={bestV:+.3f}; det.latency={mlat:.2f}s; "
              f"false alarms={n_false}; silent resets={n_reset}; "
              f"final memory {len(agent.Y)} entries")
        for i, e in enumerate(events):
            fd = " ".join(f"{PARAM_NAMES[j]}={e['cand'][j]:.2f}" for j in free)
            tag = "FALSE-ALARM" if e["false_alarm"] else (
                "FELL " if e["fell"] else "walked")
            lat = "" if e["latency"] != e["latency"] else f"lat={e['latency']:.2f}s "
            print(f"    ev{i+1:2d} t={e['onset'] if e['onset'] is not None else -1:5.1f}s "
                  f"{tag} {lat}V={e['V']:+.3f} | {fd}", flush=True)

    ev_all = [e for s in range(a.seeds) for e in all_events[s]]
    print(f"\n=== continual summary ({a.seeds} seeds) ===")
    tot = len(ev_all); tf = sum(e["fell"] for e in ev_all)
    lats = [e["latency"] for e in ev_all if e["latency"] == e["latency"]]
    print(f"  total shift events {tot}, falls {tf} ({tf/max(tot,1):.0%}); "
          f"mean detection latency {np.mean(lats) if lats else float('nan'):.2f}s; "
          f"false alarms {n_false_tot}")
    fr_e = np.mean([np.mean([e["fell"] for e in all_events[s][:max(1, len(all_events[s])//3)]])
                    for s in range(a.seeds)])
    fr_l = np.mean([np.mean([e["fell"] for e in all_events[s][-max(1, len(all_events[s])//3):]])
                    for s in range(a.seeds)])
    print(f"  mean fall rate: first third {fr_e:.0%} -> last third {fr_l:.0%} "
          f"({'LEARNING (falls drop)' if fr_l < fr_e - 1e-9 else 'no drop'})")

    csvp = os.path.join(RESULTS_DIR, "continual_events.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "event", "onset_t", "detect_t", "latency", "fell",
                    "false_alarm", "V"] + [PARAM_NAMES[i] for i in free])
        for s in range(a.seeds):
            for i, e in enumerate(all_events[s]):
                w.writerow([s, i + 1,
                            "" if e["onset"] is None else f"{e['onset']:.2f}",
                            "" if e["detect"] is None else f"{e['detect']:.2f}",
                            "" if e["latency"] != e["latency"] else f"{e['latency']:.3f}",
                            e["fell"], int(e["false_alarm"]), f"{e['V']:.4f}"]
                           + [f"{e['cand'][j]:.4f}" for j in free])
    print(f"\nsaved {csvp}")
    _plot(logs, all_events, a.seeds, free)


def _plot(logs, all_events, seeds, free):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(2, 1, figsize=(13, 8))
        lg = logs[0]
        ax[0].plot(lg["t"], lg["vx"], lw=0.7, color="tab:blue")
        ax[0].fill_between(lg["t"], -0.5, 1.2, where=lg["shift"] > 0.5,
                           color="crimson", alpha=0.10, lw=0, label="payload shifted")
        ax[0].fill_between(lg["t"], -0.5, 1.2, where=lg["state"] > 0.5,
                           color="tab:green", alpha=0.12, lw=0, label="agent responding")
        for e in all_events[0]:
            if e["onset"] is not None:
                ax[0].plot(e["onset"], 1.12, "v", color="gray", ms=6)
            if e["detect"] is not None:
                ax[0].plot(e["detect"], 1.12, "v", color="tab:orange", ms=6)
            if e["fell"]:
                ax[0].plot((e["detect"] or e["onset"]), 1.0, "x", color="red", ms=8)
        ax[0].axhline(0.5, color="k", ls=":", lw=0.8)
        ax[0].set_ylim(-0.5, 1.2); ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("vx [m/s]")
        ax[0].set_title("Seed 0: continual payload-shift recovery (pink=shifted, "
                        "green=responding, ▽ onset, ▼ detect, ✕ fall)")
        ax[0].legend(loc="upper right", fontsize=8, ncol=2); ax[0].grid(alpha=0.3)
        cols = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
        for s in range(seeds):
            ev = all_events[s]
            xs = np.arange(1, len(ev) + 1)
            Vs = [e["V"] for e in ev]
            fell = [e["fell"] for e in ev]
            ax[1].plot(xs, Vs, "-", color=cols[s % len(cols)], alpha=0.5, lw=1)
            ax[1].scatter(xs, Vs, c=["crimson" if f else cols[s % len(cols)] for f in fell],
                          s=26, label=f"seed {s}")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xlabel("shift event # (within trial)")
        ax[1].set_ylabel("post-shift V (red=fell)")
        ax[1].set_title("Per-event outcome over the trial (falls should thin out as memory fills)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "continual_recovery.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out}")
    except Exception as ex:
        print(f"(figure skipped: {ex})")


if __name__ == "__main__":
    main()
