"""Shared continual event-adaptation driver.

One long, non-episodic bout on flat ground. The robot walks; at random intervals
an EVENT recurs (payload CoM shift, or leg-actuator damage -- supplied by the
`physics` object). The event is detected by a prediction-error CUSUM (forward-
speed deficit + extra tilt vs the pre-event healthy baseline). On detection the
`responder` (one of the five arms in `event_responders`) proposes a CPG gait; it
is ramped in and held under the full event for `eval_hold` seconds:

  * if the robot FALLS   -> record V_FALL, stand it back upright and REVERT the
    event; after a random gap the event recurs;
  * if it SURVIVES the hold -> record the stability score V, revert the event,
    and the event recurs after the same random gap.

Only the physics (how the event is applied, how the robot is reset) differs
between experiments; the detection + adaptation methodology lives HERE so the
payload and damage experiments share it exactly. The physics object must provide:

    cpg = physics.setup(seed)                      # env+robot loaded, settled at origin
    st  = physics.actuate(cpg, applied, roll, pitch, frac)
                                                   # apply event at intensity frac in
                                                   # [0,1], step the sim once; returns an
                                                   # object with .base_pos (x,y,z),
                                                   # .vx (forward speed), .roll, .pitch, .fell
    cpg = physics.reset(at_xy, seed)               # stand upright at (x,y), revert event
    physics.disconnect()

`frac`=0 is the nominal (no-event) condition; `frac`=1 is the full event.
"""

from collections import namedtuple

import numpy as np

# Per-step physics readout returned by physics.actuate(): base position (x,y,z),
# forward speed vx [m/s], body roll/pitch [rad], and whether the robot has fallen.
StepState = namedtuple("StepState", ["base_pos", "vx", "roll", "pitch", "fell"])


class BoutConfig:
    """Timing / detector / event knobs for one continual bout (see run_experiment
    for the per-experiment defaults)."""

    def __init__(self, dt=0.01, duration=120.0, target_vx=0.5, v_fall=-2.0,
                 gap_min=2.0, gap_max=8.0, first_event_t=4.0,
                 eval_hold=3.0, param_ramp=30, event_ramp_t=1.0,
                 use_detector=True, detect_tau=0.4, detect_kappa=0.20,
                 detect_h=1.8, arm_frac=0.70, arm_streak_t=0.5, grace=2.5,
                 arm_timeout=6.0, detect_timeout=3.0):
        self.dt = float(dt)
        self.duration = float(duration)
        self.target_vx = float(target_vx)
        self.v_fall = float(v_fall)
        self.gap_min, self.gap_max = float(gap_min), float(gap_max)
        self.first_event_t = float(first_event_t)
        self.eval_hold = float(eval_hold)
        self.param_ramp = int(param_ramp)
        self.event_ramp_t = float(event_ramp_t)
        self.use_detector = bool(use_detector)
        self.detect_tau = float(detect_tau)
        self.detect_kappa = float(detect_kappa)
        self.detect_h = float(detect_h)
        self.arm_frac = float(arm_frac)
        self.arm_streak_t = float(arm_streak_t)
        self.grace = float(grace)
        self.arm_timeout = float(arm_timeout)
        self.detect_timeout = float(detect_timeout)


def score_V(vx, roll, pitch, target_vx, v_fall):
    """Post-event stability score over an event window (the GP-memory scale):
    saturating forward-speed reward minus an RMS-tilt penalty."""
    if len(vx) < 20:
        return v_fall
    r_v = min(max(float(np.mean(vx)), 0.0) / target_vx, 1.0)
    rms_roll = np.rad2deg(np.sqrt(np.mean(np.asarray(roll) ** 2)))
    rms_pitch = np.rad2deg(np.sqrt(np.mean(np.asarray(pitch) ** 2)))
    return float(r_v - (rms_roll + rms_pitch) / 10.0)


def _tilt_rms_deg(roll, pitch):
    if len(roll) == 0:
        return float("nan")
    return float(np.rad2deg(np.sqrt(np.mean(np.asarray(roll) ** 2
                                            + np.asarray(pitch) ** 2))))


def run_event_bout(seed, responder, physics, incumbent, cfg):
    """Run one continual bout for a single (seed, responder). Returns
    (log, events): `log` is a dict of per-step traces; `events` is a list of
    per-event outcome dicts (onset, detect, latency, fell, V, tilt_rms, dist,
    cand, false_alarm, mode)."""
    DT = cfg.dt
    incumbent = np.asarray(incumbent, float)
    rng = np.random.default_rng(777 + int(seed))

    cpg = physics.setup(seed)

    n_steps = int(round(cfg.duration / DT))
    log = {k: np.zeros(n_steps) for k in
           ("t", "y", "vx", "roll", "pitch", "shift", "state", "cusum")}
    events = []

    # ── prediction-error detector (CUSUM on a smoothed health signal) ────────
    use_det = cfg.use_detector
    a_s = DT / max(cfg.detect_tau, DT)               # EMA smoothing coefficient
    TIP_SCALE = np.deg2rad(12.0)
    vx_s = tip_s = None
    S = 0.0
    vx_base = tip_base = None
    warm_vx, warm_tip = [], []

    phase = "monitoring"                              # or "responding"
    event_active = False                              # physical event engaged
    shift_frac = 0.0                                  # 0 nominal .. 1 full event
    onset_t = None
    detect_t = None
    armed = False
    heal_streak = 0
    ARM_STREAK = int(round(cfg.arm_streak_t / DT))
    next_event_t = cfg.first_event_t
    grace_until = 1.5
    last_event_t = 0.0                                # for the force-arm watchdog
    cand = incumbent.copy()                           # last proposed gait (for logging)
    seg_start = incumbent.copy(); seg_target = incumbent.copy(); seg_anchor = 0
    applied = incumbent.copy()
    win = {"vx": [], "roll": [], "pitch": []}
    roll = pitch = 0.0
    n_false = 0
    n_reset = 0
    y_at_detect = 0.0

    def _resolve(fell, V, base_pos, k, t, tilt_rms, dist):
        """Record the outcome, revert the event, (reset upright if fell), disarm."""
        nonlocal phase, event_active, next_event_t, grace_until, cpg
        nonlocal seg_start, seg_target, seg_anchor, win, roll, pitch, S
        nonlocal onset_t, detect_t, armed, last_event_t
        # Only real event responses (onset set) update the responder's memory; a
        # false alarm evaluated a gait under the NOMINAL condition, a different
        # surface -- logged but not folded in (it would poison the memory).
        if onset_t is not None:
            responder.update(cand, V, fell)
        events.append(dict(onset=onset_t, detect=detect_t,
                           latency=(detect_t - onset_t if (detect_t is not None
                                    and onset_t is not None) else np.nan),
                           fell=int(fell), V=float(V), tilt_rms=float(tilt_rms),
                           dist=float(dist), cand=cand.copy(),
                           false_alarm=(onset_t is None), mode=responder.mode))
        if fell:
            cpg = physics.reset([base_pos[0], base_pos[1]], seed * 131 + len(events))
            roll = pitch = 0.0
            seg_start = incumbent.copy()
        else:
            seg_start = None                          # filled below from `applied`
        seg_target = incumbent.copy()
        seg_anchor = k + 1
        win = {"vx": [], "roll": [], "pitch": []}
        event_active = False                          # heal: event ramps back to nominal
        onset_t = None; detect_t = None
        armed = False
        phase = "monitoring"
        next_event_t = t + rng.uniform(cfg.gap_min, cfg.gap_max)
        grace_until = t + cfg.grace
        last_event_t = t
        S = 0.0

    k = 0
    while k < n_steps:
        t = k * DT
        # physical event schedule (responder is blind to this). Gated on the
        # detector being armed -- the robot has returned to confirmed-healthy
        # walking after the last revert -- so every event lands over a live
        # monitor and recurs "in the next gap_min..gap_max s of walking".
        if (phase == "monitoring" and not event_active and armed
                and t >= next_event_t):
            event_active = True; onset_t = t; detect_t = None
        # ramp the event intensity toward its target (1 engaged, 0 nominal)
        target_frac = 1.0 if event_active else 0.0
        step_frac = DT / max(cfg.event_ramp_t, DT)
        shift_frac = float(np.clip(shift_frac + np.sign(target_frac - shift_frac)
                                   * step_frac, 0.0, 1.0))

        frac_p = min(1.0, (k - seg_anchor) / max(1, cfg.param_ramp))
        if seg_start is None:                         # survived-heal: ramp from current
            seg_start = applied.copy()
        applied = seg_start + frac_p * (seg_target - seg_start)

        st = physics.actuate(cpg, applied, roll, pitch, shift_frac)
        base_pos, vx = st.base_pos, st.vx
        roll, pitch, fell = st.roll, st.pitch, st.fell
        tipmag = np.hypot(roll, pitch)

        vx_s = vx if vx_s is None else vx_s + a_s * (vx - vx_s)
        tip_s = tipmag if tip_s is None else tip_s + a_s * (tipmag - tip_s)
        if vx_base is None:
            if 1.5 <= t < cfg.first_event_t:
                warm_vx.append(vx); warm_tip.append(tipmag)
            if t >= cfg.first_event_t - DT:
                vx_base = max(float(np.mean(warm_vx)), 0.1) if warm_vx else 0.5
                tip_base = float(np.mean(warm_tip)) if warm_tip else 0.0

        # Arm once re-stabilized after a revert. arm_frac of the healthy baseline
        # (not ~0.85): per-stride ripple in the EMA speed otherwise keeps breaking
        # the streak and the detector never re-arms. The event deficit is far
        # larger than this margin, so detection is unaffected.
        healthy_now = (vx_base is not None and phase == "monitoring"
                       and t >= grace_until and vx_s >= cfg.arm_frac * vx_base
                       and tip_s <= tip_base + np.deg2rad(5.0))
        heal_streak = heal_streak + 1 if healthy_now else 0
        if not armed and (heal_streak >= ARM_STREAK
                          or (vx_base is not None and phase == "monitoring"
                              and t >= last_event_t + cfg.arm_timeout)):
            armed = True; S = 0.0
        e = 0.0
        if armed and phase == "monitoring":
            e = (max(0.0, vx_base - vx_s) / vx_base
                 + max(0.0, tip_s - tip_base) / TIP_SCALE)
            S = max(0.0, S + e - cfg.detect_kappa)
        log["t"][k], log["y"][k] = t, base_pos[1]
        log["vx"][k], log["roll"][k], log["pitch"][k] = vx, roll, pitch
        log["shift"][k] = shift_frac
        log["state"][k] = 1.0 if phase == "responding" else 0.0
        log["cusum"][k] = S

        if phase == "responding":
            if t - detect_t > cfg.param_ramp * DT + 0.2:
                win["vx"].append(vx); win["roll"].append(roll); win["pitch"].append(pitch)
            if fell:
                _resolve(True, cfg.v_fall, base_pos, k, t,
                         _tilt_rms_deg(win["roll"], win["pitch"]),
                         base_pos[1] - y_at_detect)
            elif t - detect_t >= cfg.param_ramp * DT + cfg.eval_hold:
                _resolve(False, score_V(win["vx"], win["roll"], win["pitch"],
                                        cfg.target_vx, cfg.v_fall),
                         base_pos, k, t,
                         _tilt_rms_deg(win["roll"], win["pitch"]),
                         base_pos[1] - y_at_detect)
        else:  # monitoring
            fired = (armed and S > cfg.detect_h) if use_det \
                else (event_active and detect_t is None)
            # liveness: if an event is engaged but the detector hasn't fired within
            # detect_timeout (a missed/limp event), force a (late) response so the
            # responder still acts and the schedule cannot stall in an engaged state.
            if (use_det and event_active and detect_t is None and not fired
                    and onset_t is not None and t - onset_t > cfg.detect_timeout):
                fired = True
            if fell and onset_t is not None:          # undetected event fall
                cand = incumbent.copy()
                _resolve(True, cfg.v_fall, base_pos, k, t, float("nan"), 0.0)
            elif fell:                                # residual nominal fall
                cpg = physics.reset([base_pos[0], base_pos[1]],
                                    seed * 131 + 90000 + k)
                roll = pitch = 0.0
                seg_start = incumbent.copy(); seg_target = incumbent.copy()
                seg_anchor = k + 1
                armed = False; heal_streak = 0; S = 0.0
                grace_until = t + cfg.grace
                n_reset += 1
            elif fired:
                detect_t = t
                y_at_detect = base_pos[1]
                if onset_t is None:
                    n_false += 1
                cand, _m = responder.propose()
                cand = np.asarray(cand, float)
                seg_start = applied.copy(); seg_target = cand.copy(); seg_anchor = k + 1
                win = {"vx": [], "roll": [], "pitch": []}
                phase = "responding"; armed = False; S = 0.0
        k += 1

    physics.disconnect()
    for key in log:
        log[key] = log[key][:k]
    return log, events, n_false, n_reset
