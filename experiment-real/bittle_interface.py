"""Bittle hardware layer for the continual payload-shift experiment.

This is the real-robot counterpart of ``PayloadPhysics`` in
``experiment-simulation/experiment-payload-adapt/run_experiment.py``: it implements
the SAME physics contract that ``methods.continual_driver.run_event_bout`` expects

    cpg = physics.setup(seed)
    st  = physics.actuate(cpg, applied, roll, pitch, frac)
    cpg = physics.reset(at_xy, seed)
    physics.disconnect()

so the detector, the responder arms and the event bookkeeping are literally the
same code on the robot as in simulation -- only what happens inside ``actuate``
changes (serial writes to a Bittle instead of ``p.stepSimulation()``).

Three things are genuinely different on hardware, and each is made explicit here
rather than papered over:

* **No foot-contact sensors.** The Righetti STOP/FAST feedback in ``JointCPG``
  needs a per-leg contact bit. Bittle has none, so by default we feed the CPG the
  contact pattern its own oscillator phase *expects* (stance <=> ``y < 0``), which
  reproduces the nominal simulated behaviour but carries no disturbance feedback.
  See ``--contacts``.
* **No odometry.** Nothing on the robot measures forward speed, so the CUSUM
  detector's speed-deficit term has nothing to feed on. With ``vx_source="none"``
  (the default) a constant nominal speed is reported, which makes that term
  identically zero and reduces the detector to its body-tilt term. Distances in
  the logs are then dead-reckoned from that constant and are NOT measurements.
* **No automatic reset.** A fallen Bittle needs a human. ``reset()`` parks the
  robot, recentres the payload and waits for the operator (or fires the firmware's
  self-right skill with ``recover="auto"``).

The payload shift itself is the rack-and-pinion CoM harness in ``printing/``:
one spare Petoi servo (the head/neck port by default) drives a mass slug along the
deck diagonal, so the driver's continuous event ramp ``frac in [0,1]`` maps
linearly onto a servo angle -- the hardware analogue of the simulated CoM offset.

CALIBRATION -- verify these on the bench before trusting a run (``--mode`` helpers
in ``run_experiment.py`` exist for each):
  1. ``--mode imu``   IMU sign/units: tilt the robot right, ``roll`` must go
                      positive; nose down, ``pitch`` must go positive. Flip with
                      ``--roll-sign`` / ``--pitch-sign``.
  2. ``--mode shift`` harness end stops: the slug must reach both extremes
                      without stalling the servo (``--shift-centered/--shift-shifted``).
  3. ``--mode walk``  the incumbent gait must actually walk (see JOINT MAPPING).
  4. ``--mode rate``  achievable serial control rate -> pick ``--dt``.

JOINT MAPPING. The 8-D gait vector is the simulator's (``methods.cpg_bounds``),
in Laikago radians. The scale factors below convert it to Bittle degrees and are
anchored so that the simulator's flat-optimal incumbent reproduces the hand-tuned
gait in ``petoi_Hopf.py``, which is known to walk on this robot:

    hip_amp  0.10 (rad, incumbent) * 120 deg/unit = 12 deg  = petoi_Hopf hip_amplitude
    knee_amp 0.50 (rad, incumbent) *  12 deg/unit =  6 deg  = petoi_Hopf knee_amplitude
    offsets                                    40 / 30 deg  = petoi_Hopf hip/knee_offset

The oscillator parameters agree to within a few percent as well (the incumbent's
w_swing/w_stance are 13.0/25.0 against the hand-tuned 12/24), so the transfer is a
rescaling of the same gait, not a leap.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):                 # vendored PetoiRobot/ + the methods/ lib
    if _p not in sys.path:
        sys.path.insert(0, _p)

from methods import continual_driver as cd
from methods.cpg_controller import JointCPG

# ── Bittle joint map ─────────────────────────────────────────────────────────
# Petoi indexes the legs clockwise from the front-left: 8/12 = front-left
# shoulder/knee, 9/13 = front-right, 10/14 = REAR-right, 11/15 = REAR-left. The
# simulator's leg order is [FL, FR, RL, RR], so the rear pair is swapped here --
# get this wrong and the trot coupling matrix drives a broken diagonal pair.
HIP_PORTS = [8, 9, 11, 10]
KNEE_PORTS = [12, 13, 15, 14]
SHIFT_PORT = 0                     # spare servo driving the CoM harness pinion

# CPG -> Bittle degrees (see JOINT MAPPING above)
HIP_OFFSET_DEG = 40.0
KNEE_OFFSET_DEG = 30.0
HIP_DEG_PER_UNIT = 120.0
KNEE_DEG_PER_UNIT = 12.0

# Servo travel guards. The serial protocol caps at +-125 deg; these are tighter,
# to keep a runaway parameter from folding a leg into the chassis.
HIP_RANGE_DEG = (-10.0, 95.0)
KNEE_RANGE_DEG = (-10.0, 95.0)

FALL_TILT_DEG = 50.0               # |roll| or |pitch| beyond this = tipped over
FALL_TICKS = 3                     # consecutive readings required (noise guard)
NOMINAL_VX = 0.08                  # [m/s] reported when no speed source exists


def _floats(text):
    """Every whitespace-separated token in `text` that parses as a float."""
    out = []
    for tok in str(text).replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


class BittleCPG(JointCPG):
    """``JointCPG`` with the Bittle joint mapping.

    The oscillators, the contact feedback and the VMC attitude feedback are
    inherited unchanged -- only the CPG-state -> joint-angle conversion is
    rescaled to Bittle degrees, by pre-scaling the two amplitude parameters and
    overriding the offsets. The attitude-feedback gains and clip are converted
    with the same knee scale factor, so the posture correction keeps the same
    authority *relative to the gait amplitude* as in simulation.
    """

    HIP_OFFSET = HIP_OFFSET_DEG
    KNEE_OFFSET = KNEE_OFFSET_DEG
    DKNEE_CLIP = JointCPG.DKNEE_CLIP * KNEE_DEG_PER_UNIT      # 0.35 rad -> 4.2 deg

    def __init__(self, n_legs=4, attitude_gain=1.0):
        super().__init__(n_legs=n_legs)
        g = float(attitude_gain) * KNEE_DEG_PER_UNIT
        self.set_gains([self.KP_ROLL * g, self.KD_ROLL * g,
                        self.KP_PITCH * g, self.KD_PITCH * g])

    def step(self, params_8d, raw_contacts, dt, roll=None, pitch=None):
        p = np.asarray(params_8d, float).copy()
        p[5] *= HIP_DEG_PER_UNIT           # hip amplitude  [rad] -> [deg]
        p[6] *= KNEE_DEG_PER_UNIT          # knee amplitude [rad] -> [deg]
        return super().step(p, raw_contacts, dt, roll=roll, pitch=pitch)

    def control_tick(self, params_8d, dt_ctrl, sub_dt, roll=None, pitch=None,
                     contacts="phase"):
        """Advance one CONTROL tick, integrating the oscillators in sub-steps.

        The serial link only sustains ~40-70 Hz, but the oscillators are an
        explicit-Euler discretization tuned at the simulator's 100 Hz: integrating
        them directly at the control period DIVERGES for a large part of the
        search box (measured: ~0/200 random gaits blow up at dt=0.01, ~103/200 at
        dt=0.025, ~191/200 at dt=0.03). NaN joint targets on a real robot are not
        an option, so the CPG is integrated at ``sub_dt`` (the simulation step)
        and only the *last* sub-step's angles are commanded.

        The attitude correction is applied ONCE, with the control period, because
        that -- not the integration step -- is how often the IMU is actually read.
        """
        n = max(1, int(round(float(dt_ctrl) / max(float(sub_dt), 1e-6))))
        h = float(dt_ctrl) / n
        hips = knees = None
        for _ in range(n):
            bits = (expected_contacts(self) if contacts == "phase"
                    else (np.ones(4, dtype=int) if contacts == "all"
                          else np.zeros(4, dtype=int)))
            hips, knees = self.step(params_8d, bits, h)      # open loop
        if self.ATTITUDE_FEEDBACK and roll is not None and pitch is not None:
            knees = knees + self.attitude_dknee(roll, pitch, float(dt_ctrl))
        return hips, knees


def expected_contacts(cpg):
    """Contact pattern the oscillator phase implies (stance <=> y < 0).

    Bittle has no foot-contact sensors; feeding the CPG its own expectation
    reproduces what the simulated controller sees on flat ground when the gait is
    working, at the cost of removing the disturbance feedback path.
    """
    return (np.asarray(cpg.y) < 0.0).astype(int)


class SyntheticRobot:
    """Caricature of a loaded Bittle, used ONLY by ``--dry-run``.

    This is not a simulator -- the PyBullet experiment is the simulator, and any
    number produced with a synthetic robot is meaningless. It exists so the
    serial-free rehearsal actually exercises the whole chain (detect -> request ->
    propose -> apply -> fall -> operator reset -> re-arm) instead of walking a
    permanently level robot that never triggers anything.

    Body tilt relaxes toward a target set by the payload offset ``frac``, reduced
    if the active gait has moved away from the incumbent -- so an adapting arm
    tips over less often than ``noadapt``, and the rehearsal covers both the
    survival and the fall branches of the driver.
    """

    def __init__(self, incumbent, box=None, seed=0, tilt_deg=62.0, tau=2.0,
                 noise_deg=1.5, adapt_credit=0.7):
        self.inc = np.asarray(incumbent, float)
        self.box = box
        self.rng = np.random.default_rng(int(seed))
        self.tilt = np.deg2rad(float(tilt_deg))
        self.tau = float(tau)
        self.noise = np.deg2rad(float(noise_deg))
        self.credit = float(adapt_credit)
        self.roll = self.pitch = 0.0

    def _adapted(self, applied):
        """How far the active gait has moved from the incumbent, in [0, 1]."""
        if self.box is None or applied is None:
            return 0.0
        lo, hi = np.asarray(self.box[0], float), np.asarray(self.box[1], float)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        d = np.abs(np.asarray(applied, float) - self.inc) / span
        return float(np.clip(np.mean(d) * 4.0, 0.0, 1.0))

    def step(self, frac, applied, dt):
        target = self.tilt * float(frac) * (1.0 - self.credit * self._adapted(applied))
        a = dt / max(self.tau, dt)
        self.roll += a * (target - self.roll) + self.noise * np.sqrt(dt) * \
            self.rng.normal()
        self.pitch += a * (0.4 * target - self.pitch) + self.noise * \
            np.sqrt(dt) * self.rng.normal()
        return self.roll, self.pitch, 0.0, ()

    def reset(self):
        self.roll = self.pitch = 0.0


class BittleLink:
    """Serial link to the robot (a thin wrapper over the vendored Petoi API).

    ``dry_run=True`` swaps in a null transport that accepts every command and
    reports a level, non-falling robot, so the whole experiment -- driver,
    responders, logging -- can be exercised without hardware.
    """

    def __init__(self, dry_run=False, imu_units="auto", roll_sign=1.0,
                 pitch_sign=1.0, keep_gyro=False):
        self.dry_run = bool(dry_run)
        self.imu_units = imu_units
        self.roll_sign = float(roll_sign)
        self.pitch_sign = float(pitch_sign)
        self.keep_gyro = bool(keep_gyro)
        self._petoi = None
        self._deg = (imu_units != "rad")   # refined by autodetect on first reads
        self._fail_streak = 0
        self.n_imu_fail = 0
        self.n_send_fail = 0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self):
        if self.dry_run:
            print("[link] dry run: no serial connection")
            return
        import PetoiRobot as petoi
        self._petoi = petoi
        # autoConnect() = connectPort + deactivate the firmware's gyro balancing,
        # which otherwise fights the CPG for the joints. Keeping it on is only
        # useful if deactivating it also silences the IMU on this firmware.
        petoi.connectPort(petoi.goodPorts)
        if not petoi.goodPorts:
            raise SystemExit("no Bittle found on any serial port -- check the USB "
                             "adapter and that no other program holds the port")
        if not self.keep_gyro:
            petoi.deacGyro()
        print(f"[link] connected: {list(petoi.goodPorts.values())}")
        time.sleep(1.0)

    def close(self):
        if self.dry_run or self._petoi is None:
            return
        try:
            self.posture("rest", delay=1.0)
        finally:
            self._petoi.closeAllSerial(self._petoi.goodPorts)
            print("[link] serial closed")

    # ── commands ─────────────────────────────────────────────────────────────
    def _send(self, task, timeout=0):
        if self.dry_run:
            return ["ok", ""]
        if self._petoi is None:            # never connected / already closed
            return -1
        res = self._petoi.send(self._petoi.goodPorts, task, timeout)
        if res == -1:
            self.n_send_fail += 1
            self._fail_streak += 1
            if self._fail_streak >= 20:
                raise SystemExit("20 consecutive serial failures -- the robot "
                                 "stopped responding (battery? unplugged?)")
        else:
            self._fail_streak = 0
        return res

    def posture(self, name, delay=1.0):
        """Fire a built-in firmware skill/posture ('balance', 'rest', 'up', ...)."""
        self._send([f"k{name}", delay])

    def joints(self, indexed):
        """Command joints as a flat [index, deg, index, deg, ...] list."""
        self._send(["I", [int(v) for v in indexed], 0])

    def read_imu(self):
        """Return (roll, pitch, yaw) in RADIANS, sign-corrected, or None.

        Petoi's 'v' token prints yaw/pitch/roll (and, on newer firmware, the raw
        accelerations). Units are firmware-dependent, so degrees vs radians is
        autodetected unless pinned with ``imu_units``.
        """
        if self.dry_run:
            return 0.0, 0.0, 0.0, ()
        raw = self._send(["v", 0], timeout=1)
        if raw == -1 or not isinstance(raw, (list, tuple)) or len(raw) < 2:
            self.n_imu_fail += 1
            return None
        vals = _floats(raw[1])
        if len(vals) < 3:
            self.n_imu_fail += 1
            return None
        yaw, pitch, roll = vals[0], vals[1], vals[2]
        if self.imu_units == "auto" and max(abs(yaw), abs(pitch), abs(roll)) > 6.5:
            self._deg = True               # radians never exceed 2*pi
        if self._deg:
            yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
        return (self.roll_sign * roll, self.pitch_sign * pitch, yaw,
                tuple(vals[3:6]))

    # ── bring-up helpers ─────────────────────────────────────────────────────
    def measure_period(self, n=60, pose=None):
        """Median wall-clock cost of one control tick (a joint write + an IMU read).

        The serial round trip -- not the CPG maths -- sets the achievable control
        rate, so ``--dt`` should be chosen from this rather than assumed."""
        pose = pose if pose is not None else self.neutral_pose()
        dts = []
        for _ in range(int(n)):
            t0 = time.perf_counter()
            self.joints(pose)
            self.read_imu()
            dts.append(time.perf_counter() - t0)
        return float(np.median(dts)), float(np.max(dts))

    @staticmethod
    def neutral_pose(shift_deg=None):
        """Standing pose in the indexed [port, deg, ...] form."""
        out = []
        for j in range(4):
            out += [HIP_PORTS[j], int(round(HIP_OFFSET_DEG)),
                    KNEE_PORTS[j], int(round(KNEE_OFFSET_DEG))]
        if shift_deg is not None:
            out += [SHIFT_PORT, int(round(shift_deg))]
        return out


class BittlePhysics:
    """The ``continual_driver`` physics contract, backed by a real Bittle.

    One instance per bout. ``frac`` (the driver's event intensity) drives the CoM
    harness servo from its centred angle to its shifted angle; a fall is detected
    from body tilt and handed to the operator.
    """

    def __init__(self, link, dt, *, shift_port=SHIFT_PORT,
                 shift_centered=0.0, shift_shifted=60.0, manual_shift=False,
                 imu_every=1, contacts="phase", attitude=True, attitude_gain=1.0,
                 fall_tilt_deg=FALL_TILT_DEG, fall_ticks=FALL_TICKS,
                 vx_source="none", nominal_vx=NOMINAL_VX, acc_axis=1,
                 recover="manual", recover_skill="up", recover_pause=5.0,
                 settle_t=1.5, cpg_dt=0.01, synthetic=None):
        self.link = link
        self.dt = float(dt)          # control period (serial-limited)
        self.cpg_dt = float(cpg_dt)  # oscillator integration step (see control_tick)
        self.shift_port = int(shift_port)
        self.shift_centered = float(shift_centered)
        self.shift_shifted = float(shift_shifted)
        self.manual_shift = bool(manual_shift)
        self.imu_every = max(1, int(imu_every))
        self.contacts = contacts
        self.attitude = bool(attitude)
        self.attitude_gain = float(attitude_gain)
        self.fall_tilt = np.deg2rad(float(fall_tilt_deg))
        self.fall_ticks = int(fall_ticks)
        self.vx_source = vx_source
        self.nominal_vx = float(nominal_vx)
        self.acc_axis = int(acc_axis)
        self.recover = recover
        self.recover_skill = recover_skill
        self.recover_pause = float(recover_pause)
        self.settle_t = float(settle_t)
        self.synthetic = synthetic          # --dry-run stand-in for the robot

        self.k = 0                     # control ticks since setup
        self.roll = self.pitch = self.yaw = 0.0
        self.vx = 0.0
        self.y = 0.0                   # dead-reckoned forward distance [m]
        self.tilt_streak = 0
        self.fallen = False
        self.last_cmd = np.zeros(8)    # commanded [hip, knee] x 4, radians
        self.n_falls = 0
        self.n_clipped = 0
        self.n_diverged = 0
        self.reset_time = 0.0          # wall seconds spent in operator recoveries
        self._shift_announced = False
        self.t_start = None

    # ── physics contract ─────────────────────────────────────────────────────
    def setup(self, seed):
        self.link.connect()
        self._park()
        self.t_start = time.perf_counter()
        return self._new_cpg()

    def actuate(self, cpg, applied, roll, pitch, frac):
        """One control tick: command the harness + the gait, then read the IMU."""
        att = (roll, pitch) if self.attitude else (None, None)
        hips, knees = cpg.control_tick(np.asarray(applied, float), self.dt,
                                       self.cpg_dt, roll=att[0], pitch=att[1],
                                       contacts=self.contacts)
        if not (np.all(np.isfinite(hips)) and np.all(np.isfinite(knees))):
            # A diverged oscillator must never reach the servos. Park the robot
            # and report a fall so the driver ends the event and resets with a
            # fresh CPG (see control_tick on why this should no longer happen).
            self.n_diverged += 1
            print(f"\n[!] CPG produced non-finite joint targets for gait "
                  f"{np.round(np.asarray(applied, float), 3).tolist()} -- parking "
                  f"and treating this as a fall", flush=True)
            self._park()
            self.fallen = True
            return cd.StepState(base_pos=(0.0, self.y, 0.0), vx=0.0,
                                roll=self.roll, pitch=self.pitch, fell=True,
                                vy=0.0, joint_angles=self.last_cmd.copy())

        cmd = []
        for j in range(4):
            h = self._clip(hips[j], HIP_RANGE_DEG)
            k = self._clip(knees[j], KNEE_RANGE_DEG)
            cmd += [HIP_PORTS[j], int(round(h)), KNEE_PORTS[j], int(round(k))]
            self.last_cmd[2 * j] = np.deg2rad(h)
            self.last_cmd[2 * j + 1] = np.deg2rad(k)
        if self.shift_port >= 0 and not self.manual_shift:
            cmd += [self.shift_port, int(round(self._shift_deg(frac)))]
        else:
            self._announce_shift(frac)
        self.link.joints(cmd)

        # The fall verdict only advances on a FRESH attitude reading, so
        # --imu-every > 1 cannot manufacture a streak out of one stale sample.
        if self.k % self.imu_every == 0:
            self._read_state(frac, applied)
            self.fallen = self._fall_check()
        self.k += 1
        self.y += self.vx * self.dt
        return cd.StepState(base_pos=(0.0, self.y, 0.0), vx=self.vx,
                            roll=self.roll, pitch=self.pitch, fell=self.fallen,
                            vy=0.0, joint_angles=self.last_cmd.copy())

    def reset(self, at_xy, seed):
        """Fall recovery: park, recentre the payload, get the robot back upright."""
        t_reset = time.perf_counter()
        self.n_falls += 1
        elapsed = time.perf_counter() - (self.t_start or time.perf_counter())
        print(f"\n[fall #{self.n_falls} at t={elapsed:6.1f}s] "
              f"roll={np.rad2deg(self.roll):+6.1f} deg  "
              f"pitch={np.rad2deg(self.pitch):+6.1f} deg", flush=True)
        self._park()                               # also recentres the harness
        if self.manual_shift:
            print("\a>>> RECENTRE the payload slug, stand the robot upright, "
                  "clear the arena.", flush=True)
        if self.recover == "auto":
            self.link.posture(self.recover_skill, delay=2.0)
            time.sleep(self.recover_pause)
        else:
            try:
                input(">>> press ENTER when the robot is upright and ready ")
            except EOFError:                       # unattended run: just wait
                time.sleep(self.recover_pause)
        self.link.posture("balance", delay=1.0)
        time.sleep(self.settle_t)
        if self.synthetic is not None:
            self.synthetic.reset()
        self.roll = self.pitch = 0.0
        self.tilt_streak = 0
        self.fallen = False
        self.vx = 0.0
        # the driver's clock does not advance while the operator works, so this is
        # tracked separately rather than counted as control-loop time
        self.reset_time += time.perf_counter() - t_reset
        return self._new_cpg()

    def disconnect(self):
        try:
            self._park()
        finally:
            self.link.close()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _new_cpg(self):
        cpg = BittleCPG(n_legs=4, attitude_gain=self.attitude_gain)
        cpg.ATTITUDE_FEEDBACK = self.attitude
        return cpg

    def _clip(self, deg, rng):
        lo, hi = rng
        if deg < lo or deg > hi:
            self.n_clipped += 1
        return float(np.clip(deg, lo, hi))

    def _shift_deg(self, frac):
        f = float(np.clip(frac, 0.0, 1.0))
        return self.shift_centered + f * (self.shift_shifted - self.shift_centered)

    def _announce_shift(self, frac):
        """Operator-in-the-loop payload shift (no harness servo fitted)."""
        if frac > 0.5 and not self._shift_announced:
            print("\a>>> SHIFT the payload slug NOW", flush=True)
            self._shift_announced = True
        elif frac < 0.05 and self._shift_announced:
            self._shift_announced = False

    def _read_state(self, frac=0.0, applied=None):
        if self.synthetic is not None:
            imu = self.synthetic.step(frac, applied, self.dt * self.imu_every)
        else:
            imu = self.link.read_imu()
        if imu is None:                            # keep the last good reading
            return
        self.roll, self.pitch, self.yaw, acc = imu
        if self.vx_source == "imu" and len(acc) > self.acc_axis:
            # leaky-integrated body acceleration: uncalibrated and drifty, but
            # enough to see the robot stall. The leak is what keeps it bounded.
            a = float(acc[self.acc_axis])
            self.vx = 0.97 * self.vx + self.dt * a
        elif self.vx_source != "imu":
            self.vx = self.nominal_vx              # constant -> tilt-only detector

    def _fall_check(self):
        tipped = (abs(self.roll) > self.fall_tilt or abs(self.pitch) > self.fall_tilt)
        self.tilt_streak = self.tilt_streak + 1 if tipped else 0
        return self.tilt_streak >= self.fall_ticks

    def _park(self):
        """Neutral stance with the payload centred (safe between-events state)."""
        self.link.joints(self.link.neutral_pose(
            shift_deg=self.shift_centered if self.shift_port >= 0 else None))
        time.sleep(0.5)
