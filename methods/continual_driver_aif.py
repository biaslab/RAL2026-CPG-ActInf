"""In-process continual driver for the unified AIF agent (aif_recovery).

Like methods.continual_driver.run_event_bout, but for an agent that needs a
per-step 100 Hz observation and OWNS its trigger. The fast part -- the MARX belief
update and the cross-entropy trigger -- runs IN the sim loop every step. The slow
part -- the GP fit + EFE proposal -- is dispatched to a BACKGROUND THREAD when the
trigger fires, so the loop keeps stepping (and, under real-time pacing, keeps its
100 Hz clock) while the optimizer computes. The proposed gait is applied when the
thread delivers it, so the optimizer's wall-clock cost shows up as sim-step latency
(`compute_latency`) exactly as it does for the out-of-process arms in the async
driver -- the robot walks on the un-adapted gait until the new gait lands.

Under real-time pacing the loop spends most of each 10 ms tick asleep (GIL free),
so the propose thread makes progress concurrently; a thread (not a process) is used
because the MARX belief must be updated in-process every step and cannot go through
an out-of-process queue at 100 Hz.

`sim_speed<=0` disables pacing (free-run). Same per-event log/CSV schema as the
async driver so the same analysis + run_experiment aggregation apply.
"""

import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from methods.continual_driver import score_V, _tilt_rms_deg


def run_event_bout_aif(seed, agent, physics, incumbent, cfg):
    """One continual bout for the in-process AIF `agent`, real-time paced
    (cfg.sim_speed) with the GP proposal computed off-thread. Returns
    (log, events, n_false, n_reset) with the async driver's schema."""
    try:
        import torch
        torch.set_num_threads(1)          # keep each bout's GP fit single-core
    except Exception:
        pass

    DT = cfg.dt
    incumbent = np.asarray(incumbent, float)
    rng = np.random.default_rng(777 + int(seed))
    tail_n = max(1, int(round(cfg.eval_hold / DT)))

    cpg = physics.setup(seed)
    n_steps = int(round(cfg.duration / DT))
    log = {k: np.zeros(n_steps) for k in
           ("t", "y", "vx", "vy", "roll", "pitch", "shift", "state", "cusum",
            "cxent", "surprise", "cum_falls",
            "ce_vy", "ce_pitch", "ce_roll", "ce_trace")}
    events = []

    state = "healthy"
    requested = proposed = False          # propose dispatched / applied this event
    event_id = 0                          # guards a stale (late) proposal
    req_event = -1
    request_t = None
    compute_latency = float("nan")
    shift_frac = 0.0
    onset_t = detect_t = None
    next_event_t = cfg.first_event_t
    grace_until = 1.5
    cand = incumbent.copy()
    seg_start = incumbent.copy(); seg_target = incumbent.copy(); seg_anchor = 0
    applied = incumbent.copy()
    win = {"vx": [], "vy": [], "roll": [], "pitch": []}
    roll = pitch = 0.0
    last_y = np.zeros(4)
    n_reset = 0
    cum_falls = 0
    y_at_onset = 0.0

    pool = ThreadPoolExecutor(max_workers=1)
    pending = None

    period = (DT / cfg.sim_speed) if cfg.sim_speed and cfg.sim_speed > 0 else 0.0
    next_wall = time.perf_counter()

    def _y_obs():
        if len(win["vx"]) >= 5:
            return np.array([np.mean(win["vx"][-tail_n:]), np.mean(win["vy"][-tail_n:]),
                             np.mean(win["pitch"][-tail_n:]), np.mean(win["roll"][-tail_n:])])
        return last_y.copy()

    def _drain_pending():
        nonlocal pending
        if pending is not None:
            try:
                pending.result()
            except Exception:
                pass
            pending = None

    def _record_event(fell, V, tilt_rms, dist):
        events.append(dict(
            onset=onset_t, detect=detect_t,
            latency=(detect_t - onset_t if (detect_t is not None
                     and onset_t is not None) else np.nan),
            fell=int(fell), V=float(V), tilt_rms=float(tilt_rms),
            dist=float(dist), cand=cand.copy(), false_alarm=False,
            mode=agent.mode, request_t=request_t, compute_latency=compute_latency))

    try:
        for k in range(n_steps):
            t = k * DT

            # Re-engage the event on schedule. `grace_until` delays engagement so
            # the robot can re-stabilize after a heal, BUT a hard liveness timeout
            # (arm_timeout past next_event_t) forces engagement regardless -- else a
            # degenerate state that keeps triggering silent resets would advance
            # grace_until forever and the event would never recur (schedule stall).
            engage_ready = (t >= next_event_t
                            and (t >= grace_until
                                 or t >= next_event_t + cfg.arm_timeout))
            if state == "healthy" and getattr(agent, "armed", True) and engage_ready:
                state = "damaged"; onset_t = t; detect_t = None
                requested = False; proposed = False; event_id += 1
                request_t = None; compute_latency = float("nan")
                y_at_onset = log["y"][k - 1] if k else 0.0
                cand = incumbent.copy()
                win = {"vx": [], "vy": [], "roll": [], "pitch": []}

            target_frac = 1.0 if state == "damaged" else 0.0
            step_frac = DT / max(cfg.event_ramp_t, DT)
            shift_frac = float(np.clip(shift_frac + np.sign(target_frac - shift_frac)
                                       * step_frac, 0.0, 1.0))

            frac_p = min(1.0, (k - seg_anchor) / max(1, cfg.param_ramp))
            applied = seg_start + frac_p * (seg_target - seg_start)

            st = physics.actuate(cpg, applied, roll, pitch, shift_frac)
            base_pos, vx = st.base_pos, st.vx
            roll, pitch, fell = st.roll, st.pitch, st.fell
            vy = st.vy
            joints = st.joint_angles if st.joint_angles is not None else np.zeros(8)

            # ── fast in-loop active-inference update + trigger ───────────────
            agent.observe([vx, vy, pitch, roll], joints)
            last_y = np.array([vx, vy, pitch, roll])

            # ── apply a gait the propose thread has delivered (non-blocking) ─
            if requested and not proposed and pending is not None and pending.done():
                try:
                    c = np.asarray(pending.result()[0], float)
                except Exception:
                    c = None
                pending = None
                if c is not None and req_event == event_id:
                    cand = c
                    seg_start = applied.copy(); seg_target = cand.copy()
                    seg_anchor = k + 1
                    proposed = True
                    detect_t = t                  # the gait becomes active now
                    compute_latency = t - request_t if request_t is not None else np.nan
                    win = {"vx": [], "vy": [], "roll": [], "pitch": []}

            log["t"][k], log["y"][k] = t, base_pos[1]
            log["vx"][k], log["roll"][k], log["pitch"][k] = vx, roll, pitch
            log["shift"][k] = shift_frac
            log["state"][k] = 1.0 if (state == "damaged" and proposed) else 0.0
            log["cusum"][k] = agent.S
            log["cxent"][k] = agent.H
            log["surprise"][k] = agent.surprise
            log["cum_falls"][k] = cum_falls
            log["vy"][k] = vy
            log["ce_vy"][k] = agent.ce_mean[1]
            log["ce_pitch"][k] = agent.ce_mean[2]
            log["ce_roll"][k] = agent.ce_mean[3]
            log["ce_trace"][k] = agent.ce_trace

            if state == "damaged" and proposed and detect_t is not None \
                    and t - detect_t > cfg.param_ramp * DT + 0.2:
                win["vx"].append(vx); win["vy"].append(vy)
                win["roll"].append(roll); win["pitch"].append(pitch)

            # ── agent-owned trigger fires -> dispatch propose OFF-THREAD ──────
            if state == "damaged" and not requested:
                fired = agent.should_fire()
                if onset_t is not None and t - onset_t > cfg.detect_timeout:
                    fired = True                  # liveness: never miss a limp
                if fired:
                    request_t = t
                    req_event = event_id
                    pending = pool.submit(agent.propose)
                    requested = True

            # ── fall handling ────────────────────────────────────────────────
            if fell and state == "damaged":
                _drain_pending()                  # finish any in-flight propose
                cum_falls += 1
                log["cum_falls"][k] = cum_falls
                agent.update(cand, _y_obs(), True)
                _record_event(True, cfg.v_fall,
                              _tilt_rms_deg(win["roll"][-tail_n:], win["pitch"][-tail_n:]),
                              base_pos[1] - y_at_onset)
                cpg = physics.reset([base_pos[0], base_pos[1]],
                                    seed * 131 + len(events))
                roll = pitch = 0.0
                seg_start = incumbent.copy(); seg_target = incumbent.copy()
                seg_anchor = k + 1
                state = "healthy"
                next_event_t = t + rng.uniform(cfg.gap_min, cfg.gap_max)
                grace_until = t + cfg.grace
                onset_t = None; detect_t = None
                requested = False; proposed = False; event_id += 1
                request_t = None; compute_latency = float("nan")
                agent.on_reset()
            elif fell:
                _drain_pending()
                cpg = physics.reset([base_pos[0], base_pos[1]],
                                    seed * 131 + 90000 + k)
                roll = pitch = 0.0
                seg_start = incumbent.copy(); seg_target = incumbent.copy()
                seg_anchor = k + 1
                grace_until = t + cfg.grace
                requested = False; proposed = False; event_id += 1
                n_reset += 1
                agent.on_reset()

            # ── real-time pacing (loop mostly sleeps -> propose thread runs) ─
            if period > 0.0:
                next_wall += period
                sleep_for = next_wall - time.perf_counter()
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
                elif sleep_for < -period:
                    next_wall = time.perf_counter()

        if state == "damaged" and proposed and len(win["vx"]) >= 20:
            V = score_V(win["vx"][-tail_n:], win["roll"][-tail_n:],
                        win["pitch"][-tail_n:], cfg.target_vx, cfg.v_fall)
            agent.update(cand, _y_obs(), False)
            _record_event(False, V,
                          _tilt_rms_deg(win["roll"][-tail_n:], win["pitch"][-tail_n:]),
                          log["y"][-1] - y_at_onset)
    finally:
        pool.shutdown(wait=False)
        physics.disconnect()

    return log, events, 0, n_reset
