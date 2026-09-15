"""Bench test of the CPG + VMC attitude control loop, robot ELEVATED ON A STAND.

This is the bring-up step between ``run_experiment.py --mode walk`` (does the
incumbent gait transfer at all?) and a real bout: it runs the SAME control loop
the experiment runs -- ``BittleCPG.control_tick`` sub-stepped at ``--cpg-dt``,
plus the VMC attitude correction applied once per control tick -- but with

  * NO payload harness motion (the shift servo is never commanded),
  * NO event, NO detector, NO responder: the gait vector is constant,
  * NO fall logic (on a stand the robot cannot fall, and a tilted stand would
    otherwise trip the tilt threshold),

so the only things under test are the oscillators, the joint mapping, and the
attitude feedback path. Feet off the ground means a wrong sign or a runaway gain
costs nothing, which is exactly why the sign check belongs here and not in a
bout.

WHAT TO ACTUALLY DO ON THE STAND
--------------------------------
0. ``--mode sign`` first, and it is not optional. It walks you through holding
   the robot in two known attitudes and reports whether the IMU agrees, because
   NOTHING downstream can tell you that: every other check regresses the
   correction against the attitude the controller was given, so an inverted IMU
   passes them all while driving the robot over on the floor.

1. ``--mode still`` next. The oscillators are amplitude-zeroed, so the legs sit
   at the neutral stance and the ONLY thing that moves the knees is the posture
   correction. Tilt the stand (or the robot in it) and watch:

       bank RIGHT (right side down)  -> the RIGHT legs (FR, RR) EXTEND,
                                        the LEFT legs (FL, RL) fold
       nose UP                       -> the FRONT legs fold,
                                        the REAR legs extend

   Both are the leveling response: the legs on the sinking corner push it back
   up. If the motion is inverted, the IMU sign is wrong for that axis -- go back
   to ``--mode sign``. The end-of-run report checks the correction against the
   leveling law numerically, but only as far as the attitude the controller was
   HANDED (see the caveat it prints).

   Two smoothers run in both this script and ``run_experiment.py`` (same
   constants, so the bench test is the loop the experiment runs): ``--imu-lpf``
   low-passes the attitude BEFORE the knee correction, because walking vibration
   puts several degrees of noise on the IMU while the correction is only ~4 deg
   wide; and ``--cmd-hys`` adds hysteresis at the whole-degree rounding boundary,
   so a target sitting on a .5 stops dithering between two degrees every tick.
   Set either to 0 to see the unsmoothed behaviour. If the legs still jitter with
   ``--no-attitude``, the feedback is not the source -- look at the control rate
   and ``--cpg-dt`` instead.

2. ``--mode walk`` next: the gait runs and the correction is superimposed. Watch
   for the correction fighting the gait (legs stuttering at the swing/stance
   transition) and for clipped joint commands, both reported at the end.

3. ``--inject roll`` if you want the loop exercised without touching the robot:
   a synthetic sinusoidal attitude is fed to the controller in place of the IMU,
   so the compute -> command -> servo path can be verified even when the IMU is
   noisy or silent. The real IMU is still read and logged alongside.

IMU SIGN CONVENTION (this is the one the controller was validated in)
---------------------------------------------------------------------
``methods.marxefe_optimizer.get_observation`` unpacks the simulated attitude as
``pitch, roll, yaw = getEulerFromQuaternion(...)`` with the robot walking in +Y,
which makes

    roll  > 0  <=>  banking RIGHT (right side down)
    pitch > 0  <=>  nose UP

and ``JointCPG.ROLL_SIGN/PITCH_SIGN = -1, -1`` were validated against THAT. Feed
the loop a pitch that is positive nose-DOWN and the pitch channel drives the
robot over instead of leveling it. (Note this contradicts the pitch line printed
by ``run_experiment.py --mode imu`` in earlier revisions -- the convention above
is the one the gains were fit in.)

The mass shifter is homed at the start of every run (``--shift-home``, default
-45 deg): the harness rests about 45 deg off the trunk axis, so left alone the
slug lies ACROSS the walking direction and its CoM offset is in the wrong axis
for the whole run. Flip the sign if it comes out perpendicular; ``none`` leaves
the servo untouched.

Output: ``results/standtest_<mode>_<stamp>.npz`` with the full per-tick trace
(attitude in, attitude measured, per-leg correction, commanded joint angles, CPG
state), for plotting in a notebook.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bittle_interface as bi
import run_experiment as rx           # incumbent + the shared link constructor

RESULTS_DIR = os.path.join(_HERE, "results")

# Attitude excursion below which the sign check has nothing to work with: a
# hand-tilt of a couple of degrees is indistinguishable from IMU noise.
MIN_EXCURSION_DEG = 3.0


def mode_sign(a):
    """Guided physical check of the IMU direction -- the one thing the rest of
    this script CANNOT establish.

    Everything downstream regresses the knee correction against the attitude the
    controller was handed, so it is consistent by construction whichever way the
    IMU points. The only ground truth is the operator holding the robot in a
    known attitude, which is safe to do precisely because it is on a stand.

    Two attitudes are held (right side down, nose up) against a level reference,
    so a constant mounting bias cancels. Cross-axis motion is reported too: this
    firmware's 'v' token is unpacked positionally as (yaw, pitch, roll), and a
    firmware that orders them differently shows up here as a tilt about one axis
    appearing on the other.
    """
    link = rx.make_link(a)
    link.connect()
    try:
        link.joints(link.neutral_pose())

        def sample(prompt, n=40):
            print(f"\n  {prompt}")
            try:
                input("  hold it there and press ENTER ")
            except EOFError:
                time.sleep(2.0)
            r, p = [], []
            for _ in range(n):
                imu = link.read_imu()
                if imu is not None:
                    r.append(imu[0])
                    p.append(imu[1])
                time.sleep(0.02)
            if len(r) < n // 4:
                raise SystemExit(
                    "  the IMU returned almost nothing. On some firmware the "
                    "gyro switch also silences it -- retry with --keep-gyro.")
            if len(set(zip(r, p))) == 1:
                raise SystemExit(
                    "  the IMU is FROZEN: every read returned the identical "
                    "sample, so the module is off and 'v' is echoing its last "
                    "one. Power-cycle the robot -- this does not clear without "
                    "a reboot. (Without this check it looks like you simply did "
                    "not tilt far enough.)")
            return np.rad2deg(np.mean(r)), np.rad2deg(np.mean(p)), len(r)

        print("\n  IMU SIGN CHECK. Hold the robot by hand (or tilt the stand); "
              "the legs stay parked.\n  The convention the attitude feedback was "
              "validated in is:\n    roll  > 0  banking RIGHT (right side down)\n"
              "    pitch > 0  nose UP")
        r0, p0, _ = sample("Hold the robot LEVEL.")
        print(f"    level reference: roll {r0:+6.1f}  pitch {p0:+6.1f} deg"
              + ("   (a large offset here is a mounting bias -- it is subtracted "
                 "below, but the controller does NOT subtract it: roll is fed in "
                 "raw)" if max(abs(r0), abs(p0)) > 5 else ""))
        r1, p1, _ = sample("Now bank the robot RIGHT SIDE DOWN, about 20 deg.")
        r2, p2, _ = sample("Now level again, then pitch the NOSE UP, about 20 deg.")

        d_roll, x_roll = r1 - r0, p1 - p0        # response, cross-axis response
        d_pitch, x_pitch = p2 - p0, r2 - r0
        print(f"\n  right side down: roll {d_roll:+6.1f} deg "
              f"(pitch moved {x_roll:+6.1f})")
        print(f"  nose up        : pitch {d_pitch:+6.1f} deg "
              f"(roll moved {x_pitch:+6.1f})")

        ok = True
        for axis, resp, cross, flag, cur in (
                ("roll", d_roll, x_roll, "--roll-sign", a.roll_sign),
                ("pitch", d_pitch, x_pitch, "--pitch-sign", a.pitch_sign)):
            if abs(resp) < MIN_EXCURSION_DEG * 2:
                ok = False
                print(f"  !! {axis}: only {abs(resp):.1f} deg of response -- "
                      f"either you did not tilt it far enough, or this axis is "
                      f"not being read at all")
            elif resp < 0:
                ok = False
                print(f"  !! {axis} is INVERTED. Run everything (this script AND "
                      f"run_experiment.py) with {flag} {-cur:g}")
            else:
                print(f"  ok {axis}: sign is correct with {flag} {cur:g}")
            if abs(cross) > abs(resp):
                ok = False
                print(f"  !! tilting about {axis} moved the OTHER axis more "
                      f"({abs(cross):.1f} vs {abs(resp):.1f} deg): this firmware "
                      f"does not report (yaw, pitch, roll) in that order, so the "
                      f"unpack in BittleLink.read_imu is wrong for it")
        print("\n  " + ("all good -- carry these flags into every run."
                        if ok else "fix the above BEFORE running on the floor: "
                                   "with a wrong sign the posture loop pushes "
                                   "the robot over instead of catching it."))
    finally:
        link.close()


def _opt_float(v):
    """A float, or None for 'none'/'off' -- lets --shift-home opt out entirely."""
    return None if str(v).strip().lower() in ("none", "off") else float(v)


def build_cpg(a):
    cpg = bi.BittleCPG(n_legs=4, attitude_gain=a.attitude_gain,
                       hip_offset=a.hip_offset, knee_offset=a.knee_offset)
    cpg.ATTITUDE_FEEDBACK = not a.no_attitude
    return cpg


def injected_attitude(a, t):
    """Synthetic (roll, pitch) in radians, or None to use the IMU."""
    if a.inject == "none":
        return None
    amp = np.deg2rad(a.inject_amp)
    w = 2.0 * np.pi * a.inject_freq
    if a.inject == "roll":
        return amp * np.sin(w * t), 0.0
    if a.inject == "pitch":
        return 0.0, amp * np.sin(w * t)
    return amp * np.sin(w * t), amp * np.sin(w * t + 0.5 * np.pi)   # both


def gait_vector(a, incumbent, t):
    """Constant gait, amplitude-ramped so the servos are not slammed at t=0.

    ``--mode still`` zeroes the two amplitude dims (hip 5, knee 6), which parks
    the legs at the neutral stance while the oscillators keep running -- so the
    knees move only under the attitude correction.
    """
    p = np.asarray(incumbent, float).copy()
    if a.mode == "still":
        p[5] = p[6] = 0.0
        return p
    ramp = 1.0 if a.ramp <= 0 else float(np.clip(t / a.ramp, 0.0, 1.0))
    p[5] *= ramp
    p[6] *= ramp
    return p


# ── the loop ─────────────────────────────────────────────────────────────────
def run(a, incumbent, dt):
    link = rx.make_link(a)
    cpg = build_cpg(a)
    n = int(round(a.duration / dt))

    print(f"\n  {a.mode} test: {a.duration:g}s at dt={dt:g}s ({1 / dt:.0f} Hz), "
          f"{n} ticks, CPG sub-stepped at {a.cpg_dt:g}s")
    print(f"  gait   : {np.round(incumbent, 3).tolist()}"
          f"{'  (amplitudes zeroed: still)' if a.mode == 'still' else ''}")
    hip_sweep, knee_lift = bi.travel_degrees(incumbent)
    print(f"  travel  : hip +-{hip_sweep:.1f} deg about {cpg.HIP_OFFSET:g}, "
          f"knee lift {knee_lift:.1f} deg from {cpg.KNEE_OFFSET:g} "
          f"(swing only)")
    stride = bi.stride_period(incumbent)
    tps = stride / dt
    print(f"  stride  : {stride:.3f}s ({1 / stride:.2f} Hz) -> {tps:.1f} commands "
          f"per stride" + ("" if tps >= bi.TICKS_PER_STRIDE_MIN else
                           "   !! see below"))
    if tps < bi.TICKS_PER_STRIDE_MIN:
        w_slow = stride * (bi.TICKS_PER_STRIDE_MIN / tps)
        print(f"    !! the servos hold the last command, so this stride is "
              f"executed as a {tps:.0f}-step staircase.\n"
              f"       This is NOT smoothable -- it is the control rate against "
              f"the gait frequency.\n"
              f"       Either slow the gait (scale w_swing/w_stance by "
              f"{stride / w_slow:.2f} for a {w_slow:.2f}s stride) or raise the "
              f"rate\n       (--imu-every 2 buys the IMU read back; the link "
              f"itself is firmware-bound).")
    print(f"  smoothing: attitude low pass "
          + (f"tau={a.imu_lpf:g}s (alpha={bi.lpf_alpha(a.imu_lpf, dt):.2f})"
             if a.imu_lpf > 0 else "OFF")
          + f", command hysteresis "
          + (f"+-{a.cmd_hys:g} deg" if a.cmd_hys > 0 else "OFF"))
    print(f"  attitude feedback: "
          f"{'OFF (open loop)' if a.no_attitude else f'ON, gains x{a.attitude_gain:g} -> kp_roll={cpg.kp_roll:.2f}, kp_pitch={cpg.kp_pitch:.2f} knee-deg/rad, clip +-{cpg.DKNEE_CLIP:.1f} deg'}")
    if a.inject != "none":
        print(f"  attitude SOURCE: injected {a.inject} sine, "
              f"{a.inject_amp:g} deg at {a.inject_freq:g} Hz "
              f"(the IMU is still read and logged, but not fed to the loop)")
    print("\n  Tilt the stand and watch the knees:\n"
          "    right side DOWN -> RIGHT legs (FR, RR) EXTEND\n"
          "    nose UP         -> FRONT legs (FL, FR) FOLD\n"
          "  Ctrl-C to stop early; the trace is still saved.\n")

    T = np.zeros(n)
    RM, PM = np.zeros(n), np.zeros(n)          # measured attitude (rad)
    RI, PI = np.zeros(n), np.zeros(n)          # attitude fed to the controller
    DK = np.zeros((n, 4))                      # per-leg knee correction (deg)
    HIP, KNEE = np.zeros((n, 4)), np.zeros((n, 4))       # commanded (deg)
    CX, CY = np.zeros((n, 4)), np.zeros((n, 4))          # oscillator state
    FRESH = np.zeros(n, dtype=bool)            # was the IMU read this tick?

    RS, PS = np.zeros(n), np.zeros(n)          # the low pass's INPUT (rad):
                                               # differs from *_meas only under
                                               # --inject

    n_clipped = n_quant = 0
    roll = pitch = 0.0
    att_f = None                               # filtered attitude state
    alpha = bi.lpf_alpha(a.imu_lpf, dt)
    quant = bi.CommandQuantizer(a.cmd_hys)
    k = 0
    link.connect()
    t_wall0 = time.perf_counter()
    try:
        # The harness rests ~45 deg off the trunk axis, so unless it is homed the
        # slug starts across the walking direction and its CoM offset is in the
        # wrong axis for the whole run.
        link.joints(link.neutral_pose(shift_deg=a.shift_home,
                                      hip_deg=a.hip_offset,
                                      knee_deg=a.knee_offset))
        if a.shift_home is not None:
            print(f"  slug homed: servo {bi.SHIFT_PORT} -> {a.shift_home:+g} deg "
                  f"(flip the sign if it ends up across the walking direction, "
                  f"not along it)")
        time.sleep(a.settle)
        t_wall0 = time.perf_counter()
        for k in range(n):
            t = k * dt
            deadline = t_wall0 + (k + 1) * dt

            if k % a.imu_every == 0:
                imu = link.read_imu()
                if imu is not None:
                    roll, pitch, _yaw, _acc = imu
                FRESH[k] = imu is not None
            inj = injected_attitude(a, t)
            r_raw, p_raw = (roll, pitch) if inj is None else inj
            # Smooth what the CONTROLLER sees; the raw values still go to the
            # log and the sign check, so filtering cannot fake a passing test.
            if alpha >= 1.0:
                r_in, p_in = r_raw, p_raw
            else:
                att_f = (r_raw, p_raw) if att_f is None else (
                    att_f[0] + alpha * (r_raw - att_f[0]),
                    att_f[1] + alpha * (p_raw - att_f[1]))
                r_in, p_in = att_f

            params = gait_vector(a, incumbent, t)
            # Same composition as BittleCPG.control_tick, but with the two terms
            # kept apart so the correction can be logged: the oscillators are
            # sub-stepped open loop, then ONE attitude correction is added with
            # the control period (which is the real IMU sampling period).
            hips, knees = cpg.control_tick(params, dt, a.cpg_dt,
                                           roll=None, pitch=None,
                                           contacts=a.contacts)
            dk = (cpg.attitude_dknee(r_in, p_in, dt) if cpg.ATTITUDE_FEEDBACK
                  else np.zeros(4))
            knees_cmd = knees + dk

            if not (np.all(np.isfinite(hips)) and np.all(np.isfinite(knees_cmd))):
                print(f"\n[!] non-finite joint target at t={t:.2f}s -- parking. "
                      f"The oscillators diverged: lower --cpg-dt or check the "
                      f"gait vector.", flush=True)
                break

            cmd, hip_deg, knee_deg = [], np.zeros(4), np.zeros(4)
            for j in range(4):
                h = float(np.clip(hips[j], *bi.HIP_RANGE_DEG))
                kk = float(np.clip(knees_cmd[j], *bi.KNEE_RANGE_DEG))
                n_clipped += int(hips[j] != h) + int(knees_cmd[j] != kk)
                hip_deg[j], knee_deg[j] = h, kk
                cmd += [bi.HIP_PORTS[j], quant(bi.HIP_PORTS[j], h),
                        bi.KNEE_PORTS[j], quant(bi.KNEE_PORTS[j], kk)]
            link.joints(cmd)
            # the servos take whole degrees, so a correction smaller than the
            # quantum is commanded but never executed -- worth knowing, since the
            # clip is only ~4 deg wide
            if np.any(np.abs(dk) > 1e-6) and np.all(
                    np.round(knee_deg) == np.round(knee_deg - dk)):
                n_quant += 1

            T[k], RM[k], PM[k], RI[k], PI[k] = t, roll, pitch, r_in, p_in
            RS[k], PS[k] = r_raw, p_raw
            DK[k], HIP[k], KNEE[k] = dk, hip_deg, knee_deg
            CX[k], CY[k] = cpg.x, cpg.y

            if k % max(1, int(0.2 / dt)) == 0:
                print(f"\r  t {t:6.1f}s | roll {np.rad2deg(r_in):+6.1f} "
                      f"pitch {np.rad2deg(p_in):+6.1f} deg | dknee "
                      + " ".join(f"{s}{v:+5.1f}" for s, v in
                                 zip(("FL", "FR", "RL", "RR"), dk))
                      + f" | clip {n_clipped:4d}   ", end="", flush=True)
            if a.pace:
                slack = deadline - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
        k += 1
    except KeyboardInterrupt:
        print("\n  stopped by the operator")
    finally:
        wall = time.perf_counter() - t_wall0
        try:
            link.joints(link.neutral_pose(shift_deg=a.shift_home,
                                      hip_deg=a.hip_offset,
                                      knee_deg=a.knee_offset))
            time.sleep(0.3)
        finally:
            link.close()

    trace = dict(t=T[:k], roll_meas=RM[:k], pitch_meas=PM[:k], roll_in=RI[:k],
                 pitch_in=PI[:k], roll_src=RS[:k], pitch_src=PS[:k],
                 dknee=DK[:k], hip_deg=HIP[:k],
                 knee_deg=KNEE[:k], cpg_x=CX[:k], cpg_y=CY[:k], fresh=FRESH[:k])
    stats = dict(wall=wall, ticks=k, n_clipped=n_clipped, n_quant=n_quant,
                 n_held=quant.n_held, n_cmds=quant.n_total,
                 n_imu_fail=link.n_imu_fail, n_send_fail=link.n_send_fail,
                 n_imu_stale=link.n_imu_stale, max_imu_stale=link.max_imu_stale)
    return trace, stats, cpg


# ── after the run ────────────────────────────────────────────────────────────
def sign_check(trace, cpg):
    """Did the correction come out the way the leveling law says?

    Regresses the per-leg knee correction on the attitude THE CONTROLLER SAW --
    so it checks the computed path (gains, per-leg geometry, that the attitude
    reaches the loop at all and is not swallowed by the clip), NOT whether the
    IMU points the way the robot is actually leaning. An inverted IMU passes
    this check; only ``--mode sign`` and your eyes can catch that.
    The pitch channel acts on the deviation from a slow EMA baseline (so a steady
    incline is tolerated rather than fought), so the same EMA is reproduced here
    -- regressing on raw pitch would understate the pitch gain for slow tilts.
    """
    roll, pitch, dk = trace["roll_in"], trace["pitch_in"], trace["dknee"]
    if len(roll) < 20:
        return None
    ema = np.zeros_like(pitch)
    e = pitch[0]
    for i, p in enumerate(pitch):
        e += cpg.ATT_EMA_ALPHA * (p - e)
        ema[i] = e
    d_pitch = pitch - ema

    A = np.column_stack([roll, d_pitch, np.ones_like(roll)])
    out = {"roll_range": float(np.ptp(np.rad2deg(roll))),
           "pitch_range": float(np.ptp(np.rad2deg(pitch))), "legs": {}}
    unclipped = np.all(np.abs(dk) < 0.98 * cpg.DKNEE_CLIP, axis=1)
    if unclipped.sum() < 20:              # saturated throughout: use everything
        unclipped = np.ones(len(roll), dtype=bool)
    for j, name in enumerate(("FL", "FR", "RL", "RR")):
        coef, *_ = np.linalg.lstsq(A[unclipped], dk[unclipped, j], rcond=None)
        # Expected slopes in commanded knee-deg per deg of attitude. The gain
        # MAGNITUDE is used, so a negative --attitude-gain (which inverts the
        # whole loop) is flagged rather than quietly inverting the expectation.
        exp_r = cpg._LEFT[j] * cpg.ROLL_SIGN * abs(cpg.kp_roll) * np.deg2rad(1.0)
        exp_p = cpg._FRONT[j] * cpg.PITCH_SIGN * abs(cpg.kp_pitch) * np.deg2rad(1.0)
        out["legs"][name] = dict(
            roll_slope=float(coef[0] * np.deg2rad(1.0)), roll_expected=float(exp_r),
            pitch_slope=float(coef[1] * np.deg2rad(1.0)), pitch_expected=float(exp_p))
    return out


def report(a, trace, stats, cpg, dt):
    k, wall = stats["ticks"], stats["wall"]
    print(f"\n\n  == stand test: {a.mode}, {k} ticks in {wall:.1f}s ==")
    if k == 0:
        print("  nothing ran")
        return
    hz, target = k / max(wall, 1e-9), 1.0 / dt
    print(f"  control rate   {hz:5.1f} Hz against a target of {target:.0f} Hz"
          + ("" if hz >= 0.95 * target else
             "  !! the link cannot sustain --dt; the gait ran SLOWER than "
             "commanded (re-measure with `run_experiment.py --mode rate`)"))
    print(f"  joint travel   hip {trace['hip_deg'].min():6.1f} .. "
          f"{trace['hip_deg'].max():6.1f} deg   knee "
          f"{trace['knee_deg'].min():6.1f} .. {trace['knee_deg'].max():6.1f} deg "
          f"(limits {bi.HIP_RANGE_DEG} / {bi.KNEE_RANGE_DEG})")
    if stats["n_clipped"]:
        print(f"  !! {stats['n_clipped']} commands hit the safety limits -- the "
              f"gait asks for more travel than allowed")
    if stats["n_imu_fail"]:
        print(f"  !! {stats['n_imu_fail']} IMU reads failed (last good value "
              f"reused); {100 * trace['fresh'].mean():.0f}% of ticks had a fresh "
              f"reading")
    if stats.get("max_imu_stale", 0) > 20:
        print(f"  !! the IMU repeated the same sample {stats['max_imu_stale']} "
              f"times in a row: it FROZE during the run, and every reading after "
              f"that was the last real one. Power-cycle the robot.")
    if stats["n_send_fail"]:
        print(f"  !! {stats['n_send_fail']} serial writes failed")

    dk = trace["dknee"]
    if not a.no_attitude:
        sat = float(np.mean(np.abs(dk) > 0.98 * cpg.DKNEE_CLIP))
        print(f"  correction     |dknee| mean {np.abs(dk).mean():4.2f} deg, max "
              f"{np.abs(dk).max():4.2f} deg, at the +-{cpg.DKNEE_CLIP:.1f} deg "
              f"clip for {100 * sat:.0f}% of ticks")
        if stats.get("n_cmds"):
            print(f"  command hysteresis changed "
                  f"{100 * stats['n_held'] / stats['n_cmds']:.1f}% of the "
                  f"{stats['n_cmds']} servo writes (only those where plain "
                  f"rounding would have flipped a degree -- that is the dither "
                  f"it removed)")
        if stats["n_quant"]:
            print(f"  !! on {100 * stats['n_quant'] / k:.0f}% of ticks the "
                  f"correction was smaller than the servos' 1 deg quantum and "
                  f"was rounded away (the clip is only {cpg.DKNEE_CLIP:.1f} deg "
                  f"wide -- raise --attitude-gain if this dominates)")

        chk = sign_check(trace, cpg)
        print(f"\n  -- correction path (against the attitude the loop was "
              f"HANDED: roll spanned {chk['roll_range']:.1f} deg, pitch "
              f"{chk['pitch_range']:.1f} deg) --")
        if max(chk["roll_range"], chk["pitch_range"]) < MIN_EXCURSION_DEG:
            print(f"     the robot never moved more than {MIN_EXCURSION_DEG:g} "
                  f"deg: tilt the stand during the run (or use --inject roll), "
                  f"otherwise this check is fitting noise")
        else:
            print("     leg   d(knee)/d(roll)      d(knee)/d(pitch)   "
                  "[commanded deg per deg]")
            bad = []
            for name, v in chk["legs"].items():
                marks = []
                for axis, span in (("roll", chk["roll_range"]),
                                   ("pitch", chk["pitch_range"])):
                    got, exp = v[f"{axis}_slope"], v[f"{axis}_expected"]
                    ok = span < MIN_EXCURSION_DEG or np.sign(got) == np.sign(exp)
                    marks.append("  " if span < MIN_EXCURSION_DEG else
                                 ("ok" if ok else "!!"))
                    if not ok:
                        bad.append(axis)
                print(f"     {name}   {v['roll_slope']:+6.3f} (want "
                      f"{v['roll_expected']:+6.3f}) {marks[0]}   "
                      f"{v['pitch_slope']:+6.3f} (want "
                      f"{v['pitch_expected']:+6.3f}) {marks[1]}")
            if bad:
                axes = sorted(set(bad))
                print(f"     !! the {'/'.join(axes)} channel is INVERTED with "
                      f"respect to the leveling law. Check --attitude-gain "
                      f"({a.attitude_gain:g}) and the gains in JointCPG -- this "
                      f"is a software mismatch, not an IMU one (the IMU sign "
                      f"cancels out of this fit).")
            else:
                print("     signs agree with the leveling law on all four legs.")
            print("     (magnitudes below the expectation are normal: the fit "
                  "sees the D term and the clip as well as the P term.)")
        print("\n     This says nothing about which way the IMU points: it "
              "regresses the correction on the attitude the loop was given, so "
              "an inverted IMU passes it. Establish that with --mode sign, and "
              "confirm with your eyes -- banking the robot right must EXTEND "
              "the right legs, nose UP must FOLD the front legs.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(RESULTS_DIR, f"standtest_{a.mode}_{stamp}.npz")
    meta = dict(mode=a.mode, dt=dt, cpg_dt=a.cpg_dt, contacts=a.contacts,
                imu_lpf=a.imu_lpf, cmd_hys=a.cmd_hys, shift_home=a.shift_home,
                attitude=not a.no_attitude, attitude_gain=a.attitude_gain,
                gains=[cpg.kp_roll, cpg.kd_roll, cpg.kp_pitch, cpg.kd_pitch],
                dknee_clip=cpg.DKNEE_CLIP, roll_sign=a.roll_sign,
                pitch_sign=a.pitch_sign, inject=a.inject,
                inject_amp=a.inject_amp, inject_freq=a.inject_freq,
                imu_every=a.imu_every, dry_run=a.dry_run,
                params=np.asarray(rx.load_incumbent(a.incumbent_json)).tolist(),
                rate_hz=hz, **{k_: v for k_, v in stats.items() if k_ != "wall"})
    np.savez_compressed(path, meta=json.dumps(meta, default=float), **trace)
    print(f"\n  saved {path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knee-lift", type=float, default=None,
                    help="[deg] knee rotation during SWING -- how far the foot "
                         "is picked up (incumbent: 8.5). The knee term is active "
                         "in swing only, so this is the ground-clearance knob; "
                         "raise it if the robot drags or scuffs")
    ap.add_argument("--hip-sweep", type=float, default=None,
                    help="[deg] hip half-sweep about its offset, i.e. stride "
                         "length (incumbent: 17.0)")
    ap.add_argument("--knee-offset", type=float, default=None,
                    help=f"[deg] stance knee angle (default "
                         f"{bi.KNEE_OFFSET_DEG:g}). Raising it extends the legs "
                         f"and lifts the trunk -- the other half of 'it is not "
                         f"clearing the ground' when a payload squashes the "
                         f"servos. The park pose follows it")
    ap.add_argument("--hip-offset", type=float, default=None,
                    help=f"[deg] stance hip angle (default "
                         f"{bi.HIP_OFFSET_DEG:g})")
    ap.add_argument("--stride", type=float, default=None,
                    help="[s] stretch the gait to this stride period by scaling "
                         "w_swing and w_stance together (the swing/stance ratio "
                         "is preserved). The serial link caps the control rate, "
                         "so a fast stride is commanded as a coarse staircase; "
                         "this is the knob that buys commands per stride")
    ap.add_argument("--imu-lpf", type=float, default=0.06,
                    help="[s] time constant of the low pass on the attitude fed "
                         "to the knee correction (0 disables). Walking vibration "
                         "puts several degrees of noise on the IMU and the "
                         "correction is only ~4 deg wide, so unfiltered it "
                         "spends its budget chasing noise")
    ap.add_argument("--cmd-hys", type=float, default=0.25,
                    help="[deg] hysteresis at the whole-degree rounding "
                         "boundary, so a target sitting on a .5 does not dither "
                         "between two degrees every tick (0 disables)")
    ap.add_argument("--shift-home", type=_opt_float, default=-45.0,
                    help="[deg] commanded to the mass-shifter servo before the "
                         "run so the slug lies ALONG the walking direction (the "
                         "harness rests ~45 deg off-axis). Flip the sign if it "
                         "ends up across instead; 'none' leaves it untouched")
    ap.add_argument("--mode", choices=["sign", "still", "walk"], default="walk",
                    help="'sign': guided IMU direction check, no gait -- run "
                         "this FIRST, nothing else can establish it. 'still': "
                         "amplitudes zeroed, so only the posture correction "
                         "moves the knees. 'walk': the incumbent gait with the "
                         "correction superimposed")
    ap.add_argument("--duration", type=float, default=30.0, help="[s]")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds held at the neutral stance before starting")
    ap.add_argument("--ramp", type=float, default=1.0,
                    help="gait amplitudes ramped in over this many seconds, so "
                         "the servos are not slammed at t=0 ('walk' only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no robot: null serial transport, level IMU. Combine "
                         "with --inject to exercise the whole loop")
    ap.add_argument("--incumbent-json", default=None,
                    help="per-robot gait (default results/incumbent.json)")

    g = ap.add_argument_group("robot / timing")
    g.add_argument("--dt", type=float, default=None,
                   help="control period [s]; measured at startup if omitted")
    g.add_argument("--cpg-dt", type=float, default=0.01,
                   help="oscillator integration sub-step [s] -- keep it at the "
                        "simulation step: the explicit-Euler discretization "
                        "diverges for much of the search box above ~0.025 s")
    g.add_argument("--calib-ticks", type=int, default=60)
    g.add_argument("--no-pace", dest="pace", action="store_false",
                   help="do not sleep to hold --dt (let the serial link set the "
                        "rate, as the experiment does)")
    g.add_argument("--imu-every", type=int, default=1)
    rx.add_wifi_args(g)
    g.add_argument("--imu-units", choices=["auto", "deg", "rad"], default="auto")
    g.add_argument("--keep-gyro", action="store_true",
                   help="do NOT deactivate the firmware's gyro balancing (it "
                        "otherwise fights the CPG for the joints)")
    g.add_argument("--roll-sign", type=float, default=1.0,
                   help="+-1 into the convention roll>0 = banking right")
    g.add_argument("--pitch-sign", type=float, default=1.0,
                   help="+-1 into the convention pitch>0 = nose UP")
    g.add_argument("--contacts", choices=["phase", "none", "all"], default="phase",
                   help="no contact sensors exist; 'phase' feeds the CPG the "
                        "pattern its own oscillator expects (what the experiment "
                        "does). 'none' is the literal truth on a stand and shows "
                        "how hard the Righetti STOP term distorts the gait")
    g.add_argument("--no-attitude", action="store_true",
                   help="open-loop reference: no posture correction at all")
    g.add_argument("--attitude-gain", type=float, default=1.0,
                   help="scales the gains transferred from simulation")

    g = ap.add_argument_group("synthetic attitude (no hands needed)")
    g.add_argument("--inject", choices=["none", "roll", "pitch", "both"],
                   default="none",
                   help="feed the loop a sinusoidal attitude instead of the IMU")
    g.add_argument("--inject-amp", type=float, default=8.0, help="[deg]")
    g.add_argument("--inject-freq", type=float, default=0.25, help="[Hz]")
    a = ap.parse_args()

    if a.mode == "sign":
        return mode_sign(a)

    incumbent = rx.load_incumbent(a.incumbent_json)
    if a.stride is not None:
        # Rescaling both oscillator frequencies keeps the swing/stance RATIO --
        # the shape of the gait -- and only stretches it in time, which is the
        # one knob that buys commands per stride without touching the geometry.
        now = bi.stride_period(incumbent)
        f = now / float(a.stride)
        incumbent = np.asarray(incumbent, float).copy()
        incumbent[1] *= f
        incumbent[2] *= f
        print(f"[gait] stride stretched {now:.3f}s -> {a.stride:g}s "
              f"(w_swing/w_stance x{f:.2f}); this is a DIFFERENT gait from the "
              f"simulated incumbent -- save it to results/incumbent.json if it "
              f"walks")
    if a.hip_sweep is not None or a.knee_lift is not None:
        was = bi.travel_degrees(incumbent)
        incumbent = bi.set_travel(incumbent, hip_deg=a.hip_sweep,
                                  knee_deg=a.knee_lift)
        now = bi.travel_degrees(incumbent)
        print(f"[gait] travel {was[0]:.1f}/{was[1]:.1f} -> {now[0]:.1f}/"
              f"{now[1]:.1f} deg (hip half-sweep / knee lift); a DIFFERENT gait "
              f"from the simulated incumbent -- save it to "
              f"results/incumbent.json if it walks")
    dt = a.dt
    if dt is None:
        link = rx.make_link(a)
        link.connect()
        try:
            med, worst, t_w, t_f = link.amortised_period(
                imu_every=a.imu_every, n=a.calib_ticks)
        finally:
            link.close()
        # 5 ms grid, not 10: at a ~34 ms tick the coarse grid threw away 16% of
        # the achievable rate, and the rate is what the gait is starved of.
        dt = max(0.01, float(np.ceil(med * 200) / 200))
        print(f"[timing] average tick {1000 * med:.1f} ms (worst "
              f"{1000 * worst:.1f} ms) -> dt = {dt:g} s ({1 / dt:.0f} Hz)")
        if a.imu_every > 1:
            print(f"[timing]   write {1000 * t_w:.1f} ms, write+read "
                  f"{1000 * t_f:.1f} ms, IMU read every {a.imu_every} ticks")

    trace, stats, cpg = run(a, incumbent, dt)
    report(a, trace, stats, cpg, dt)


if __name__ == "__main__":
    main()
