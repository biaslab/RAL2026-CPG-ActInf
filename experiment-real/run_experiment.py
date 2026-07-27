"""Continual payload-shift adaptation on the REAL robot (Petoi Bittle).

The hardware counterpart of
``experiment-simulation/experiment-payload-adapt/run_experiment.py``. It runs the
SAME driver (``methods.continual_driver.run_event_bout``), the SAME event-detector
and the SAME responder arms; only the physics is swapped for a serial link to a
Bittle carrying the rack-and-pinion CoM harness from ``printing/`` (see
``bittle_interface.py`` for the joint mapping and the hardware caveats).

One bout = one long, non-episodic walk:

  * the robot walks on the incumbent (simulation flat-optimal) gait;
  * after a few seconds the harness servo slides the payload slug along the deck
    diagonal -- a persistent, asymmetric CoM offset -- ramped over ~1 s;
  * a prediction-error CUSUM on the body attitude detects the resulting limp and
    the chosen ARM proposes a new gait, which is ramped in;
  * the shift PERSISTS. Only a fall reverts it: the operator stands the robot back
    up, the slug is recentred, and after a random gap the shift re-engages.

Headline metric = FALLS PER BOUT, exactly as in simulation.

Arms: noadapt / grid / bo / esc / safegp / oracle (``methods.event_responders``)
plus ``aif``, the unified active-inference agent. ``oracle`` needs a hardware-fit
optimum in ``results/payload_optima.json`` and is unavailable until one exists.

Results are written in the simulator's schema, so
``methods.continual_analysis`` and the simulation notebooks read them unchanged:
  results/continual_events.csv   one row per event (APPENDED across sessions)
  results/continual_summary.csv  per-method aggregates, regenerated each session
  results/logs/<arm>_seed<k>.npz per-bout step traces

Bring-up (do these in order, on a fresh battery, before any real run)::

    cd experiment-real
    python run_experiment.py --mode rate            # achievable control rate -> --dt
    python run_experiment.py --mode imu             # IMU units + roll/pitch SIGNS
    python run_experiment.py --mode shift           # harness end stops
    python run_experiment.py --mode walk --duration 20   # does the incumbent walk?
    python run_experiment.py --arms noadapt --seeds 1 --duration 90   # first bout

Full session (supervised, one bout at a time, ~2 min each)::

    python run_experiment.py --arms noadapt aif safegp bo --seeds 3 --duration 120

Everything can be rehearsed without a robot -- driver, responders, logging -- with
``--dry-run``, which swaps in a null serial transport plus a synthetic body
attitude. It exercises the plumbing; the numbers it produces are meaningless.
"""

import argparse
import csv
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

from methods import continual_driver as cd
from methods import event_responders as er
from methods import responder_worker as rw

import bittle_interface as bi

RESULTS_DIR = os.path.join(_HERE, "results")
OPTIMA_JSON = os.path.join(RESULTS_DIR, "payload_optima.json")
INCUMBENT_JSON = os.path.join(RESULTS_DIR, "incumbent.json")

PARAM_NAMES = ["coupling", "w_swing", "w_stance", "F_FAST", "STOP", "hipA",
               "kneeA", "b"]

# The simulation flat-optimal incumbent, kept in sync with
# experiment-simulation/experiment-payload-adapt/run_experiment.py (folders do not
# import from each other). Override per robot with results/incumbent.json.
INCUMBENT = np.array([7.607, 13.0498, 25.0, 52.4044, 0.5, 0.1, 0.5, 10.0])

# CPG dims the responders search (same reduced space as the simulation, so the
# hardware and simulated comparisons are head-to-head).
FREE_DIMS_PAYLOAD = [0, 1, 3, 4, 7]

EVENT_COLS = ["method", "seed", "event", "onset", "detect", "latency", "fell",
              "V", "tilt_rms", "dist", "false_alarm", "request_t",
              "compute_latency", "trial_dist"]


# ── inputs ───────────────────────────────────────────────────────────────────
def load_incumbent(path=None):
    path = path or INCUMBENT_JSON
    if os.path.exists(path):
        return np.asarray(json.load(open(path))["params"], float)
    return INCUMBENT.copy()


def load_oracle_target(path=None):
    path = path or OPTIMA_JSON
    if not os.path.exists(path):
        return None
    return np.asarray(json.load(open(path))["shifted"]["params"], float)


def build_spec(arm, incumbent, box, free, oracle_target, seed, a):
    """Picklable ResponderSpec for the out-of-process responder worker (the same
    construction the simulation uses, so an arm behaves identically here)."""
    safegp_kwargs = dict(n_init=a.n_init, safe_V=a.safe_V, beta=a.beta,
                         kappa=a.kappa, objective=a.objective, r_fall=a.r_fall,
                         efe_y_star=a.efe_y_star, efe_tau2=a.efe_tau2)
    return rw.ResponderSpec(
        name=arm, incumbent=np.asarray(incumbent, float), box=box,
        free_dims=list(free), oracle_target=oracle_target, seed=int(seed),
        safegp_kwargs=safegp_kwargs, seed_fall=(arm in ("bo", "safegp")),
        v_fall=cd.BoutConfig().v_fall)


def make_link(a):
    return bi.BittleLink(dry_run=a.dry_run, imu_units=a.imu_units,
                         roll_sign=a.roll_sign, pitch_sign=a.pitch_sign,
                         keep_gyro=a.keep_gyro)


def make_physics(a, dt, incumbent=None, box=None, seed=0):
    link = make_link(a)
    # --dry-run: no robot, so a synthetic stand-in supplies the body attitude,
    # otherwise the rehearsal would walk a permanently level robot that never
    # triggers the detector. Its numbers mean nothing (see SyntheticRobot).
    synthetic = (bi.SyntheticRobot(incumbent if incumbent is not None else INCUMBENT,
                                   box, seed=seed) if a.dry_run else None)
    physics = bi.BittlePhysics(
        link, dt=dt, synthetic=synthetic,
        shift_port=(-1 if a.manual_shift else a.shift_port),
        shift_centered=a.shift_centered, shift_shifted=a.shift_shifted,
        manual_shift=a.manual_shift, imu_every=a.imu_every,
        contacts=a.contacts, cpg_dt=a.cpg_dt, attitude=not a.no_attitude,
        attitude_gain=a.attitude_gain, fall_tilt_deg=a.fall_tilt,
        vx_source=a.vx_source, acc_axis=a.acc_axis,
        # a synthetic robot cannot be picked up, so the rehearsal never waits
        recover=("auto" if a.dry_run else a.recover),
        recover_skill=a.recover_skill,
        recover_pause=(0.0 if a.dry_run else a.recover_pause),
        settle_t=(0.0 if a.dry_run else 1.5))
    return link, physics


# ── one bout ─────────────────────────────────────────────────────────────────
def run_bout(arm, seed, a, incumbent, box, free, oracle_target, dt):
    link, physics = make_physics(a, dt, incumbent, box, seed)
    cfg = cd.BoutConfig(
        dt=dt, duration=a.duration, gap_min=a.gap_min, gap_max=a.gap_max,
        event_ramp_t=a.shift_ramp_t, eval_hold=a.eval_hold,
        use_detector=not a.no_detector, detect_tau=a.detect_tau,
        detect_kappa=a.detect_kappa, detect_h=a.detect_h, grace=a.grace,
        # On hardware the serial round trip paces the loop (--dt is measured from
        # it with --mode rate), so the driver must not add sleeps on top. With no
        # robot there is nothing to pace against, so the driver's own real-time
        # clock is used instead -- without it a bout would finish in seconds and
        # the responder's proposal would arrive after the bout had ended, which is
        # not the latency the experiment is meant to measure.
        sim_speed=(a.dry_speed if a.dry_run else 0.0))

    t0 = time.perf_counter()
    if arm == "aif":
        from methods.aif_recovery import UnifiedAIFAgent
        from methods.continual_driver_aif import run_event_bout_aif
        goal_std = (0.25, 0.25, np.deg2rad(12), np.deg2rad(12))
        agent = UnifiedAIFAgent(incumbent, box, free, seed, dt=dt,
                                target_vx=cfg.target_vx, goal_std=goal_std,
                                trigger_vx_std=a.aif_trigger_vx_std)
        out = run_event_bout_aif(seed, agent, physics, incumbent, cfg)
    else:
        spec = build_spec(arm, incumbent, box, free, oracle_target, seed, a)
        out = cd.run_event_bout(seed, spec, physics, incumbent, cfg)
    log, events, n_false, n_reset = out

    # exclude operator recoveries: the driver's clock is frozen while the robot is
    # being picked up, so counting that time would flag every bout with a fall
    wall = time.perf_counter() - t0 - physics.reset_time
    # with --dry-run the driver paces itself at --dry-speed, so compare against
    # the accelerated wall-clock target rather than the nominal bout length
    nominal = cfg.duration / (a.dry_speed if a.dry_run and a.dry_speed > 0 else 1.0)
    print(f"  bout walking time {wall:.1f}s vs nominal {nominal:.1f}s "
          f"({wall / max(nominal, 1e-9):.2f}x); {physics.k} control ticks, "
          f"measured rate {physics.k / max(wall, 1e-9):.1f} Hz")
    if abs(wall - nominal) > 0.25 * nominal:
        print(f"  !! logged time (k*dt) and wall clock disagree by "
              f"{100 * abs(wall - nominal) / nominal:.0f}% -- re-measure with "
              f"--mode rate and set --dt accordingly, or event timings in the "
              f"CSV are not seconds")
    if physics.n_diverged:
        print(f"  !! {physics.n_diverged} control ticks produced non-finite joint "
              f"targets and were parked -- raise --cpg-dt resolution or check the "
              f"proposed gaits")
    if physics.n_clipped:
        print(f"  !! {physics.n_clipped} joint commands hit the safety limits "
              f"({bi.HIP_RANGE_DEG} / {bi.KNEE_RANGE_DEG} deg) -- the gait is "
              f"asking for more travel than allowed")
    if link.n_imu_fail:
        print(f"  !! {link.n_imu_fail} IMU reads failed (last good value reused)")
    return log, events


# ── results ──────────────────────────────────────────────────────────────────
def append_events(out_dir, arm, seed, events, log, free):
    cols = EVENT_COLS + [PARAM_NAMES[j] for j in free]
    path = os.path.join(out_dir, "continual_events.csv")
    new = not os.path.exists(path)
    trial_dist = float(log["y"][-1] - log["y"][0]) if len(log["y"]) else 0.0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for i, e in enumerate(events):
            row = dict(method=arm, seed=seed, event=i + 1, onset=e["onset"],
                       detect=e["detect"], latency=e["latency"], fell=e["fell"],
                       V=e["V"], tilt_rms=e["tilt_rms"], dist=e["dist"],
                       false_alarm=int(e["false_alarm"]),
                       request_t=e["request_t"],
                       compute_latency=e["compute_latency"],
                       trial_dist=trial_dist)
            for j in free:
                row[PARAM_NAMES[j]] = float(e["cand"][j])
            w.writerow({c: row.get(c, "") for c in cols})
    return path


def write_summary(out_dir):
    """Regenerate the per-method summary from every event logged so far (hardware
    sessions accumulate in the CSV across invocations)."""
    from methods import continual_analysis as ca
    rows = ca.load_events(out_dir)
    tab = ca.summary_table(rows)
    path = os.path.join(out_dir, "continual_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_seeds", "falls_per_bout",
                                          "falls_sem", "mean_tilt_surv",
                                          "mean_dist_under_fault"])
        w.writeheader()
        for m, t in tab.items():
            w.writerow(dict(method=m, n_seeds=t["n_seeds"],
                            falls_per_bout=t["falls"][0], falls_sem=t["falls"][1],
                            mean_tilt_surv=t["tilt"][0],
                            mean_dist_under_fault=t["dist"][0]))
    print("\n  == falls per bout (all sessions in this results dir) ==")
    for m, t in tab.items():
        print(f"     {m:8s} {t['falls'][0]:5.2f} +- {t['falls'][1]:.2f} "
              f"over {t['n_seeds']} bout(s); surviving tilt "
              f"{t['tilt'][0]:5.1f} deg")
    return path


# ── bring-up modes ───────────────────────────────────────────────────────────
def mode_rate(a):
    """Measure the achievable control rate (serial round trip, not the CPG)."""
    link = make_link(a)
    link.connect()
    try:
        med, worst = link.measure_period(n=a.calib_ticks)
        print(f"\n  median tick {1000 * med:6.1f} ms ({1 / med:5.1f} Hz), "
              f"worst {1000 * worst:6.1f} ms  "
              f"[joint write + IMU read, --imu-every 1]")
        print(f"  -> run with --dt {max(0.01, np.ceil(med * 100) / 100):.2f} "
              f"(round the median UP; the CPG integrates at --dt, so a dt below "
              f"the achievable period makes the gait run slower than commanded)")
        if a.imu_every > 1:
            print(f"  (--imu-every {a.imu_every} would amortise the IMU read over "
                  f"{a.imu_every} ticks)")
    finally:
        link.close()


def mode_imu(a):
    """Stream IMU readings so the operator can verify units and SIGNS."""
    link = make_link(a)
    link.connect()
    print("\n  Tilt the robot and check the signs the CPG's attitude feedback "
          "assumes:\n    roll  > 0  when the LEFT side goes UP (banking right)\n"
          "    pitch > 0  when the NOSE goes DOWN\n"
          "  If either is inverted, re-run everything with --roll-sign -1 "
          "and/or --pitch-sign -1.\n  Ctrl-C to stop.\n")
    try:
        while True:
            imu = link.read_imu()
            if imu is None:
                print("  no IMU response -- is the gyro switched off in "
                      "firmware? try --keep-gyro", flush=True)
            else:
                roll, pitch, yaw, acc = imu
                print(f"\r  roll {np.rad2deg(roll):+7.1f}  "
                      f"pitch {np.rad2deg(pitch):+7.1f}  "
                      f"yaw {np.rad2deg(yaw):+7.1f} deg   acc {acc}      ",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        link.close()


def mode_shift(a):
    """Sweep the CoM harness servo between its centred and shifted end stops."""
    link = make_link(a)
    link.connect()
    print(f"\n  sweeping joint {a.shift_port}: {a.shift_centered:g} -> "
          f"{a.shift_shifted:g} deg. Watch for the slug binding or the servo "
          f"stalling at either end; narrow the range if it does.\n")
    try:
        link.joints(link.neutral_pose(shift_deg=a.shift_centered))
        time.sleep(1.0)
        for cycle in range(3):
            for frac in list(np.linspace(0, 1, 40)) + list(np.linspace(1, 0, 40)):
                deg = a.shift_centered + frac * (a.shift_shifted - a.shift_centered)
                link.joints([a.shift_port, int(round(deg))])
                print(f"\r  cycle {cycle + 1}/3  frac {frac:4.2f}  "
                      f"servo {deg:+6.1f} deg  ", end="", flush=True)
                time.sleep(a.shift_ramp_t / 40.0)
        print()
    except KeyboardInterrupt:
        print()
    finally:
        link.close()


def mode_walk(a, incumbent, dt, box=None):
    """Walk the incumbent gait open loop -- the bring-up check that the simulated
    gait transfers at all (the ``petoi_Hopf.py`` equivalent, through the shared
    controller and the shared parameter vector)."""
    link, physics = make_physics(a, dt, incumbent, box)
    cpg = physics.setup(0)
    n = int(round(a.duration / dt))
    print(f"\n  walking the incumbent for {a.duration:g}s "
          f"({n} ticks at dt={dt:g}); Ctrl-C to stop\n")
    t0 = time.perf_counter()
    roll = pitch = 0.0
    try:
        for k in range(n):
            st = physics.actuate(cpg, incumbent, roll, pitch, 0.0)
            roll, pitch = st.roll, st.pitch
            if k % 10 == 0:
                print(f"\r  t {k * dt:6.1f}s  roll {np.rad2deg(roll):+6.1f}  "
                      f"pitch {np.rad2deg(pitch):+6.1f} deg"
                      f"{'   FALLEN' if st.fell else '        '}",
                      end="", flush=True)
            if st.fell:
                print("\n  robot fell -- stopping")
                break
    except KeyboardInterrupt:
        print()
    finally:
        wall = time.perf_counter() - t0
        print(f"\n  {physics.k} ticks in {wall:.1f}s "
              f"({physics.k / max(wall, 1e-9):.1f} Hz); "
              f"{physics.n_clipped} clipped commands")
        physics.disconnect()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["run", "walk", "imu", "shift", "rate"],
                    default="run", help="'run' is the experiment; the others are "
                                        "bring-up checks (see the docstring)")
    ap.add_argument("--arms", nargs="+", default=["noadapt"],
                    choices=er.ALL_ARMS + ["aif"],
                    help="run sequentially, one supervised bout at a time")
    ap.add_argument("--seeds", type=int, default=1,
                    help="bouts per arm (each is a separate physical run)")
    ap.add_argument("--duration", type=float, default=120.0, help="bout length [s]")
    ap.add_argument("--dry-run", action="store_true",
                    help="no robot: null serial transport plus a synthetic body "
                         "attitude, so the driver, the arms and the logging can "
                         "be rehearsed. The numbers it produces are meaningless")
    ap.add_argument("--dry-speed", type=float, default=1.0,
                    help="--dry-run only: real-time factor. 1.0 makes a bout take "
                         "its nominal wall-clock time (so responder latency is "
                         "realistic); larger is faster but inflates that latency")
    ap.add_argument("--no-prompt", action="store_true",
                    help="do not wait for the operator between bouts")
    ap.add_argument("--out-dir", default=RESULTS_DIR)

    g = ap.add_argument_group("robot / timing")
    g.add_argument("--dt", type=float, default=None,
                   help="control period [s]. Default: measured at startup "
                        "(--mode rate reports it). The CPG integrates at this "
                        "dt, so it must match the achievable serial rate")
    g.add_argument("--cpg-dt", type=float, default=0.01,
                   help="oscillator integration step [s]. The CPG is sub-stepped "
                        "at this rate within each --dt control tick: it is an "
                        "explicit-Euler discretization tuned at the simulator's "
                        "100 Hz and DIVERGES for much of the search box if "
                        "integrated directly at a 40 Hz control period")
    g.add_argument("--calib-ticks", type=int, default=60,
                   help="ticks used to measure the control period")
    g.add_argument("--imu-every", type=int, default=1,
                   help="read the IMU every N control ticks (an IMU read costs a "
                        "serial round trip; >1 buys control rate for staleness)")
    g.add_argument("--imu-units", choices=["auto", "deg", "rad"], default="auto")
    g.add_argument("--keep-gyro", action="store_true",
                   help="do NOT deactivate the firmware's gyro balancing. Only "
                        "useful if deactivating it also silences the IMU on this "
                        "firmware -- it otherwise fights the CPG for the joints")
    g.add_argument("--roll-sign", type=float, default=1.0,
                   help="+-1; verify with --mode imu (see its printout)")
    g.add_argument("--pitch-sign", type=float, default=1.0)
    g.add_argument("--contacts", choices=["phase", "none", "all"], default="phase",
                   help="Bittle has no foot-contact sensors; 'phase' feeds the "
                        "CPG the contact pattern its own oscillator expects")
    g.add_argument("--no-attitude", action="store_true",
                   help="disable the VMC attitude feedback (open-loop CPG)")
    g.add_argument("--attitude-gain", type=float, default=1.0,
                   help="scales the attitude-feedback gains about the values "
                        "transferred from simulation")
    g.add_argument("--fall-tilt", type=float, default=bi.FALL_TILT_DEG,
                   help="|roll| or |pitch| [deg] counted as a fall")
    g.add_argument("--vx-source", choices=["none", "imu"], default="none",
                   help="'none': no odometry exists, report a constant speed so "
                        "the detector reduces to its tilt term (distances in the "
                        "logs are then dead-reckoned, not measured). 'imu': "
                        "leaky-integrated body acceleration -- uncalibrated")
    g.add_argument("--acc-axis", type=int, default=1,
                   help="--vx-source imu only: which of the three reported "
                        "accelerations points forward (firmware dependent)")
    g.add_argument("--recover", choices=["manual", "auto"], default="manual",
                   help="'manual': wait for the operator after a fall; 'auto': "
                        "fire the firmware self-right skill and continue")
    g.add_argument("--recover-skill", default="up")
    g.add_argument("--recover-pause", type=float, default=5.0)

    g = ap.add_argument_group("payload harness")
    g.add_argument("--shift-port", type=int, default=bi.SHIFT_PORT,
                   help="servo index driving the rack-and-pinion CoM harness")
    g.add_argument("--shift-centered", type=float, default=0.0,
                   help="servo angle [deg] with the slug centred")
    g.add_argument("--shift-shifted", type=float, default=60.0,
                   help="servo angle [deg] at full CoM offset -- verify the end "
                        "stop with --mode shift before running")
    g.add_argument("--manual-shift", action="store_true",
                   help="no harness fitted: prompt the operator to move the "
                        "payload by hand at each event")
    g.add_argument("--shift-ramp-t", type=float, default=1.0,
                   help="ramp duration of the shift [s]")

    g = ap.add_argument_group("detector / schedule")
    g.add_argument("--no-detector", action="store_true",
                   help="idealised: respond at shift onset instead of detecting")
    g.add_argument("--detect-kappa", type=float, default=0.20)
    g.add_argument("--detect-h", type=float, default=1.8)
    g.add_argument("--detect-tau", type=float, default=0.4)
    g.add_argument("--gap-min", type=float, default=4.0,
                   help="min healthy walking between events [s] (longer than the "
                        "simulation default: the operator needs time)")
    g.add_argument("--gap-max", type=float, default=10.0)
    g.add_argument("--grace", type=float, default=2.5)
    g.add_argument("--eval-hold", type=float, default=10.0)

    g = ap.add_argument_group("responder knobs (as in simulation)")
    g.add_argument("--free-dims", type=int, nargs="+", default=None,
                   help=f"CPG dims to search (default {FREE_DIMS_PAYLOAD} = "
                        f"{[PARAM_NAMES[i] for i in FREE_DIMS_PAYLOAD]})")
    g.add_argument("--incumbent-json", default=None,
                   help="per-robot incumbent gait (default results/incumbent.json)")
    g.add_argument("--n-init", type=int, default=4)
    g.add_argument("--safe-V", type=float, default=-0.8)
    g.add_argument("--beta", type=float, default=2.5)
    g.add_argument("--kappa", type=float, default=1.5)
    g.add_argument("--objective", choices=["ucb", "efe"], default="efe")
    g.add_argument("--r-fall", type=float, default=0.22)
    g.add_argument("--efe-y-star", type=float, default=1.0)
    g.add_argument("--efe-tau2", type=float, default=0.5)
    g.add_argument("--aif-trigger-vx-std", type=float, default=1000.0)
    a = ap.parse_args()

    incumbent = load_incumbent(a.incumbent_json)
    free = a.free_dims if a.free_dims is not None else FREE_DIMS_PAYLOAD

    # ── control period: measure it unless pinned ─────────────────────────────
    dt = a.dt
    if dt is None and a.mode not in ("rate", "imu", "shift"):
        link = make_link(a)
        link.connect()
        try:
            med, worst = link.measure_period(n=a.calib_ticks)
        finally:
            link.close()
        dt = max(0.01, float(np.ceil(med * a.imu_every * 100) / 100))
        print(f"[timing] measured tick {1000 * med:.1f} ms (worst "
              f"{1000 * worst:.1f} ms) -> dt = {dt:g} s ({1 / dt:.0f} Hz)")

    if a.mode == "rate":
        return mode_rate(a)
    if a.mode == "imu":
        return mode_imu(a)
    if a.mode == "shift":
        return mode_shift(a)
    if a.mode == "walk":
        return mode_walk(a, incumbent, dt)  # box unused: no responder here

    # ── the experiment ───────────────────────────────────────────────────────
    from methods.cpg_bounds import bounds_lower, bounds_upper
    box = (bounds_lower.numpy(), bounds_upper.numpy())
    oracle_target = load_oracle_target()
    if "oracle" in a.arms and oracle_target is None:
        raise SystemExit(
            "the oracle arm needs a HARDWARE-fit post-shift optimum in "
            f"{OPTIMA_JSON}; the simulated optimum is not a valid oracle for "
            "this robot. Drop 'oracle' from --arms, or fit one on the robot "
            "first.")
    os.makedirs(a.out_dir, exist_ok=True)
    log_dir = os.path.join(a.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\ncontinual payload shift on hardware: arms={a.arms} x {a.seeds} "
          f"bout(s) x {a.duration:g}s at dt={dt:g}s")
    print(f"  incumbent: {np.round(incumbent, 3).tolist()}")
    print(f"  searching dims {free} ({[PARAM_NAMES[i] for i in free]})")
    print(f"  shift: {'operator (manual)' if a.manual_shift else f'servo {a.shift_port}: {a.shift_centered:g} -> {a.shift_shifted:g} deg'}"
          f", ramp {a.shift_ramp_t:g}s, re-engages {a.gap_min:g}-{a.gap_max:g}s "
          f"after each fall")
    det = ("idealised (respond at onset)" if a.no_detector else
           f"attitude CUSUM (kappa={a.detect_kappa}, h={a.detect_h})")
    print(f"  detector: {det}")
    if a.vx_source == "none":
        print("  NOTE --vx-source none: no odometry, so the detector is "
              "tilt-only and logged distances are dead-reckoned, not measured")

    done = []
    for arm in a.arms:
        for seed in range(a.seeds):
            print(f"\n{'=' * 70}\n  ARM {arm}   bout {seed + 1}/{a.seeds}\n{'=' * 70}")
            if not (a.no_prompt or a.dry_run):
                print("  Place the robot at the start of the arena, slug centred, "
                      "battery charged.")
                try:
                    input("  press ENTER to start the bout ")
                except EOFError:
                    pass
            try:
                log, events = run_bout(arm, seed, a, incumbent, box, free,
                                       oracle_target, dt)
            except KeyboardInterrupt:
                print("\n  bout aborted by the operator; nothing recorded for it")
                continue
            np.savez_compressed(os.path.join(log_dir, f"{arm}_seed{seed}.npz"),
                                **{k: v for k, v in log.items()})
            append_events(a.out_dir, arm, seed, events, log, free)
            nf = sum(e["fell"] for e in events)
            done.append((arm, seed, nf, len(events)))
            print(f"  [{arm} seed{seed}] {nf} fall(s) over {len(events)} event(s)")

    if done:
        write_summary(a.out_dir)
        print(f"\nsaved {os.path.join(a.out_dir, 'continual_events.csv')} "
              f"(appended)\nsaved per-bout traces in {log_dir}")


if __name__ == "__main__":
    main()
