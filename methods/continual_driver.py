"""Shared continual event-adaptation driver.

One long, non-episodic bout on flat ground. The robot walks healthy; after a gap
an EVENT engages (payload CoM shift, or leg-actuator damage -- supplied by the
`physics` object) and PERSISTS. A prediction-error CUSUM (forward-speed deficit +
extra tilt vs the pre-event healthy baseline) detects it and the `responder` (one
of the five arms in `event_responders`) proposes a CPG gait, ramped in. From then
the robot just keeps walking under the persistent event:

  * the event is ONLY reverted when the robot FALLS -- it is never auto-healed.
    On a fall: record V_FALL, fold it into the responder's memory, stand the robot
    back upright at its position, HEAL the event, and after a random gap the event
    re-engages (the next fall cycle);
  * if the responder adapts and does NOT fall, it keeps walking under the event
    for the rest of the bout (one long surviving event, scored at the end).

So the headline metric is FALLS PER BOUT: no-adapt tips over roughly every
time-to-fall seconds and racks up many falls; a good adapter falls rarely. Only
the physics (how the event is applied / the robot reset) differs between
experiments; the detection + adaptation methodology lives HERE so the payload and
damage experiments share it exactly. The physics object must provide:

    cpg = physics.setup(seed)                      # env+robot loaded, settled at origin
    st  = physics.actuate(cpg, applied, roll, pitch, frac)
                                                   # apply event at intensity frac in
                                                   # [0,1], step the sim once; returns an
                                                   # object with .base_pos (x,y,z),
                                                   # .vx (forward speed), .roll, .pitch, .fell
    cpg = physics.reset(at_xy, seed)               # stand upright at (x,y), heal the event
    physics.disconnect()

`frac`=0 is the nominal (healthy) condition; `frac`=1 is the full event.
"""

import time
from collections import namedtuple

import numpy as np

from methods.responder_worker import ResponderWorker

# Per-step physics readout returned by physics.actuate(): base position (x,y,z),
# forward speed vx [m/s], body roll/pitch [rad], and whether the robot has fallen.
# `vy` (lateral speed) and `joint_angles` (measured hip+knee positions) are extra
# fields used by the in-process AIF agent (continual_driver_aif); they default so
# existing physics/responders that ignore them keep working unchanged.
StepState = namedtuple("StepState", ["base_pos", "vx", "roll", "pitch", "fell",
                                     "vy", "joint_angles"],
                       defaults=[0.0, None])


class BoutConfig:
    """Timing / detector / event knobs for one continual bout (see run_experiment
    for the per-experiment defaults). The event PERSISTS until a fall; `eval_hold`
    is only the trailing window over which a surviving gait's stability score V is
    measured (for the responder's memory), NOT an auto-heal timer."""

    def __init__(self, dt=0.01, duration=120.0, target_vx=0.5, v_fall=-2.0,
                 gap_min=2.0, gap_max=8.0, first_event_t=4.0,
                 eval_hold=3.0, param_ramp=30, event_ramp_t=1.0,
                 use_detector=True, detect_tau=0.4, detect_kappa=0.20,
                 detect_h=1.8, arm_frac=0.70, arm_streak_t=0.5, grace=2.5,
                 arm_timeout=6.0, detect_timeout=3.0, sim_speed=1.0):
        self.dt = float(dt)
        self.duration = float(duration)
        # Real-time pacing factor: the sim loop is paced to sim_speed x 100 Hz so a
        # propose() taking T wall-seconds maps to ~T*sim_speed/dt sim steps of
        # latency. sim_speed=1.0 is physically faithful; larger runs faster but
        # inflates the effective compute penalty; <=0 disables pacing (free-run).
        self.sim_speed = float(sim_speed)
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
    """Stability score over a window (the GP-memory scale): saturating forward-
    speed reward minus an RMS-tilt penalty."""
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


def run_event_bout(seed, responder_spec, physics, incumbent, cfg):
    """Run one continual bout for a single (seed, responder). Returns
    (log, events, n_false, n_reset). `log` is a dict of per-step traces
    (incl. `cum_falls`); `events` is a list of per-event outcome dicts with
    keys: onset, detect, latency, fell, V, tilt_rms, dist, cand, false_alarm,
    mode, request_t, compute_latency. An "event" begins when the event engages
    and ends at the next fall (or at the bout end for a surviving gait).

    The responder runs OUT OF PROCESS (methods.responder_worker): the sim loop
    never blocks on propose()/update(). When the detector fires the loop posts a
    propose-request tagged with the current event id and keeps walking on the
    current gait; it applies the returned gait only once the worker delivers it,
    so the wall-clock optimize time shows up as sim-step latency (logged as
    `compute_latency`). `responder_spec` is a picklable ResponderSpec; the live
    responder object is built inside the worker."""
    DT = cfg.dt
    incumbent = np.asarray(incumbent, float)
    rng = np.random.default_rng(777 + int(seed))
    tail_n = max(1, int(round(cfg.eval_hold / DT)))   # survival-scoring window
    responder_mode = responder_spec.name              # == the arm's .mode

    worker = ResponderWorker(responder_spec)
    cpg = physics.setup(seed)

    n_steps = int(round(cfg.duration / DT))
    log = {k: np.zeros(n_steps) for k in
           ("t", "y", "vx", "roll", "pitch", "shift", "state", "cusum", "cum_falls")}
    events = []

    # ── prediction-error detector (CUSUM on a smoothed health signal) ────────
    use_det = cfg.use_detector
    a_s = DT / max(cfg.detect_tau, DT)               # EMA smoothing coefficient
    TIP_SCALE = np.deg2rad(12.0)
    vx_s = tip_s = None
    S = 0.0
    vx_base = tip_base = None
    warm_vx, warm_tip = [], []

    state = "healthy"                                 # or "damaged"
    requested = False                                 # a propose was posted this event
    proposed = False                                  # a proposal has been APPLIED
    event_id = 0                                      # tags requests; guards staleness
    request_t = None                                  # when the propose was posted
    compute_latency = float("nan")                    # apply_t - request_t [s]
    shift_frac = 0.0                                  # 0 healthy .. 1 full event
    onset_t = None                                    # when the event engaged
    detect_t = None                                   # when the responder's gait applied
    armed = False
    heal_streak = 0
    ARM_STREAK = int(round(cfg.arm_streak_t / DT))
    next_event_t = cfg.first_event_t
    grace_until = 1.5
    last_heal_t = 0.0                                 # for the force-arm watchdog
    cand = incumbent.copy()                           # gait active this event
    seg_start = incumbent.copy(); seg_target = incumbent.copy(); seg_anchor = 0
    applied = incumbent.copy()
    win = {"vx": [], "roll": [], "pitch": []}
    roll = pitch = 0.0
    n_false = 0            # (kept for signature compatibility; ~unused here)
    n_reset = 0           # silent resets from residual healthy falls
    cum_falls = 0
    y_at_onset = 0.0

    def _record_event(fell, V, base_pos, tilt_rms, dist):
        events.append(dict(
            onset=onset_t, detect=detect_t,
            latency=(detect_t - onset_t if (detect_t is not None
                     and onset_t is not None) else np.nan),
            fell=int(fell), V=float(V), tilt_rms=float(tilt_rms),
            dist=float(dist), cand=cand.copy(),
            false_alarm=False, mode=responder_mode,
            request_t=request_t, compute_latency=compute_latency))

    # real-time pacing: period between sim steps in wall-clock (0 -> free-run)
    period = (DT / cfg.sim_speed) if cfg.sim_speed and cfg.sim_speed > 0 else 0.0
    next_wall = time.perf_counter()
    try:
        for k in range(n_steps):
            t = k * DT

            # ── event schedule: engage only from confirmed-healthy walking ───
            if state == "healthy" and armed and t >= next_event_t:
                state = "damaged"; onset_t = t; detect_t = None
                requested = False; proposed = False; event_id += 1
                request_t = None; compute_latency = float("nan")
                y_at_onset = log["y"][k - 1] if k else 0.0
                cand = incumbent.copy()
                win = {"vx": [], "roll": [], "pitch": []}
                S = 0.0

            target_frac = 1.0 if state == "damaged" else 0.0
            step_frac = DT / max(cfg.event_ramp_t, DT)
            shift_frac = float(np.clip(shift_frac + np.sign(target_frac - shift_frac)
                                       * step_frac, 0.0, 1.0))

            frac_p = min(1.0, (k - seg_anchor) / max(1, cfg.param_ramp))
            applied = seg_start + frac_p * (seg_target - seg_start)

            st = physics.actuate(cpg, applied, roll, pitch, shift_frac)
            base_pos, vx = st.base_pos, st.vx
            roll, pitch, fell = st.roll, st.pitch, st.fell
            tipmag = np.hypot(roll, pitch)

            # ── apply a gait the worker has delivered (non-blocking) ─────────
            # Drain the result queue: apply a proposal matching the CURRENT event,
            # discard stale ones left over from an event that already ended.
            while True:
                got = worker.poll()
                if got is None:
                    break
                rid, cand_new, _mode = got
                if rid == event_id and requested and not proposed:
                    cand = np.asarray(cand_new, float)
                    seg_start = applied.copy(); seg_target = cand.copy()
                    seg_anchor = k + 1
                    proposed = True
                    detect_t = t                      # the gait becomes active now
                    compute_latency = (t - request_t if request_t is not None
                                       else np.nan)
                    win = {"vx": [], "roll": [], "pitch": []}

            vx_s = vx if vx_s is None else vx_s + a_s * (vx - vx_s)
            tip_s = tipmag if tip_s is None else tip_s + a_s * (tipmag - tip_s)
            if vx_base is None:
                if 1.5 <= t < cfg.first_event_t:
                    warm_vx.append(vx); warm_tip.append(tipmag)
                if t >= cfg.first_event_t - DT:
                    vx_base = max(float(np.mean(warm_vx)), 0.1) if warm_vx else 0.5
                    tip_base = float(np.mean(warm_tip)) if warm_tip else 0.0

            # ── arm the detector once re-stabilized after a heal ─────────────
            healthy_now = (vx_base is not None and state == "healthy"
                           and t >= grace_until and vx_s >= cfg.arm_frac * vx_base
                           and tip_s <= tip_base + np.deg2rad(5.0))
            heal_streak = heal_streak + 1 if healthy_now else 0
            if not armed and (heal_streak >= ARM_STREAK
                              or (vx_base is not None and state == "healthy"
                                  and t >= last_heal_t + cfg.arm_timeout)):
                armed = True; S = 0.0

            # ── CUSUM prediction error while damaged and not yet requested ───
            if state == "damaged" and not requested:
                e = (max(0.0, vx_base - vx_s) / vx_base
                     + max(0.0, tip_s - tip_base) / TIP_SCALE) if vx_base else 0.0
                S = max(0.0, S + e - cfg.detect_kappa)

            log["t"][k], log["y"][k] = t, base_pos[1]
            log["vx"][k], log["roll"][k], log["pitch"][k] = vx, roll, pitch
            log["shift"][k] = shift_frac
            log["state"][k] = 1.0 if (state == "damaged" and proposed) else 0.0
            log["cusum"][k] = S
            log["cum_falls"][k] = cum_falls

            # collect the post-response window (for survival scoring / stability)
            if state == "damaged" and proposed and detect_t is not None \
                    and t - detect_t > cfg.param_ramp * DT + 0.2:
                win["vx"].append(vx); win["roll"].append(roll)
                win["pitch"].append(pitch)

            # ── request a gait once the event is detected (non-blocking) ─────
            # The sim keeps walking on the current gait; the worker's reply is
            # applied by the drain block above whenever it arrives, so a slow
            # optimizer costs real sim-time (compute_latency), not a frozen loop.
            if state == "damaged" and not requested:
                fired = (S > cfg.detect_h) if use_det else True
                if (use_det and onset_t is not None
                        and t - onset_t > cfg.detect_timeout):
                    fired = True                      # liveness: never miss a limp
                if fired:
                    request_t = t
                    worker.request_propose(event_id)
                    requested = True

            # ── fall handling ────────────────────────────────────────────────
            # `cand` is the gait ACTUALLY active at the fall: the applied proposal,
            # or the incumbent if the robot fell before the worker's reply landed
            # (the realistic penalty a slow method pays). Bumping event_id discards
            # any still-in-flight proposal for this now-ended event.
            if fell and state == "damaged":           # event fall -> heal + reset
                cum_falls += 1
                log["cum_falls"][k] = cum_falls
                worker.push_update(cand, cfg.v_fall, True)
                _record_event(True, cfg.v_fall, base_pos,
                              _tilt_rms_deg(win["roll"][-tail_n:], win["pitch"][-tail_n:]),
                              base_pos[1] - y_at_onset)
                cpg = physics.reset([base_pos[0], base_pos[1]],
                                    seed * 131 + len(events))
                roll = pitch = 0.0
                seg_start = incumbent.copy(); seg_target = incumbent.copy()
                seg_anchor = k + 1
                state = "healthy"; armed = False; heal_streak = 0; S = 0.0
                next_event_t = t + rng.uniform(cfg.gap_min, cfg.gap_max)
                grace_until = t + cfg.grace
                last_heal_t = t
                onset_t = None; detect_t = None
                requested = False; proposed = False; event_id += 1
                request_t = None; compute_latency = float("nan")
            elif fell:                                # residual healthy fall
                cpg = physics.reset([base_pos[0], base_pos[1]],
                                    seed * 131 + 90000 + k)
                roll = pitch = 0.0
                seg_start = incumbent.copy(); seg_target = incumbent.copy()
                seg_anchor = k + 1
                armed = False; heal_streak = 0; S = 0.0
                grace_until = t + cfg.grace
                last_heal_t = t
                requested = False; proposed = False; event_id += 1
                n_reset += 1

            # ── real-time pace so wall-clock optimize time maps to sim steps ─
            if period > 0.0:
                next_wall += period
                sleep_for = next_wall - time.perf_counter()
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
                elif sleep_for < -period:             # far behind: resync, no burst
                    next_wall = time.perf_counter()

        # ── bout end: score a surviving gait (event that never fell) ─────────
        if state == "damaged" and proposed and len(win["vx"]) >= 20:
            V = score_V(win["vx"][-tail_n:], win["roll"][-tail_n:],
                        win["pitch"][-tail_n:], cfg.target_vx, cfg.v_fall)
            worker.push_update(cand, V, False)
            _record_event(False, V, [log["y"][-1]] * 3,
                          _tilt_rms_deg(win["roll"][-tail_n:], win["pitch"][-tail_n:]),
                          log["y"][-1] - y_at_onset)
    finally:
        worker.close()
        physics.disconnect()

    return log, events, n_false, n_reset
