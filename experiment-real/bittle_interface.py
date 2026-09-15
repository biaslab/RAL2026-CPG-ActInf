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
  needs a per-leg contact bit. Bittle has none, so by default (``--contacts
  phase``) we feed the CPG the contact pattern its own oscillator phase *expects*
  (stance <=> ``y < 0``).

  Be clear about what that does. The expectation agrees with the phase BY
  CONSTRUCTION, so the STOP branch fires on essentially every tick and the FAST
  branch never fires, leaving a permanent
  ``s_i = STOP*(w_i*x_i - coupling_y_i)``. Folded into the y-update that is
  ``(1+STOP)`` on the frequency term and ``(1-STOP)`` on the coupling term --
  at the incumbent ``STOP = 0.5``, a standing 1.5x on w and 0.5x on gamma. So
  the robot is not running the gait its parameter vector literally names; it is
  running a consistently gain-scheduled version of it.

  Dropping the term instead (``--contacts none``) is NOT the safer option: it was
  measured in simulation on 2026-09-15 at ~33.6 falls per 300 s bout against 5.18
  with the term present, i.e. locomotion stops working. Until the robot has real
  foot switches, ``phase`` is the right default and the rescaling above is simply
  part of the hardware controller's definition.
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
  1. ``--mode imu``   IMU sign/units: bank the robot right side DOWN, ``roll``
                      must go positive; nose UP, ``pitch`` must go positive
                      (that is the convention ``get_observation`` feeds the
                      simulated gains). Flip with ``--roll-sign``/``--pitch-sign``.
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
import socket
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

WIFI_HOST_AP = "192.168.4.1"       # dongle's own AP (sketch_wifidongle.ino)
WIFI_PORT = 23                     # raw TCP<->Serial bridge port
WIFI_TIMEOUT = 1.0                 # read timeout [s]; matches the USB port's

FALL_TILT_DEG = 50.0               # |roll| or |pitch| beyond this = tipped over
FALL_TICKS = 3                     # consecutive readings required (noise guard)
NOMINAL_VX = 0.08                  # [m/s] reported when no speed source exists


LIMIT_CYCLE_R = np.sqrt(JointCPG.U)      # |x|,|y| peak on the limit cycle


def travel_degrees(params_8d):
    """(hip half-sweep, knee lift) in degrees for a gait vector.

    The oscillator rides a limit cycle of radius sqrt(U), so the hip sweeps
    +-hip_amp*sqrt(U) about its offset and -- because the knee term is
    ``KNEE_OFFSET - knee_amp * max(0, y)``, i.e. active in SWING ONLY -- the foot
    lifts by knee_amp*sqrt(U) of knee rotation. These are the numbers to think in
    when a robot is dragging its feet; the raw parameters are in the simulator's
    units and hide a factor of sqrt(2).
    """
    return (float(params_8d[5]) * HIP_DEG_PER_UNIT * LIMIT_CYCLE_R,
            float(params_8d[6]) * KNEE_DEG_PER_UNIT * LIMIT_CYCLE_R)


def set_travel(params_8d, hip_deg=None, knee_deg=None):
    """Gait vector with the hip sweep / knee lift set to the given degrees."""
    p = np.asarray(params_8d, float).copy()
    if hip_deg is not None:
        p[5] = float(hip_deg) / (HIP_DEG_PER_UNIT * LIMIT_CYCLE_R)
    if knee_deg is not None:
        p[6] = float(knee_deg) / (KNEE_DEG_PER_UNIT * LIMIT_CYCLE_R)
    return p


def stride_period(params_8d):
    """Seconds per full oscillator cycle for a gait vector.

    The Hopf oscillator spends half a cycle at ``w_swing`` and half at
    ``w_stance`` (params 1 and 2), so the period is the sum of the two half
    periods. Divided by the control period this gives COMMANDS PER STRIDE, which
    is the number that decides whether a gait looks smooth on hardware: the
    servos hold the last commanded angle, so a stride covered by a handful of
    ticks is executed as a staircase no filtering can smooth.
    """
    w_sw, w_st = float(params_8d[1]), float(params_8d[2])
    return np.pi / max(w_sw, 1e-6) + np.pi / max(w_st, 1e-6)


TICKS_PER_STRIDE_MIN = 20     # below this the staircase is visible; ~40 is smooth


def lpf_alpha(tau, dt):
    """EMA weight for a first-order low pass of time constant ``tau`` [s].

    ``tau <= 0`` disables the filter (alpha = 1, i.e. pass the input through).
    """
    tau = float(tau)
    return 1.0 if tau <= 0.0 else float(dt) / (tau + float(dt))


class CommandQuantizer:
    """Whole-degree servo commands, with hysteresis at the rounding boundary.

    The servos take integers. A target sitting near a .5 boundary therefore
    flips between two degrees on IMU noise alone -- audible as buzz, visible as
    jitter -- while accomplishing nothing: the correction being chased is
    *smaller than the quantum* (see ``stand_test.py``'s n_quant, which counts
    exactly this). Holding the previous command until the target has moved clear
    of the boundary costs at most ``hys`` degrees of steady-state accuracy and
    removes the dither.

    This is deliberately NOT a low-pass on the joint command: the gait's own
    sweep must not be slowed down, only the sub-degree hunting suppressed.
    """

    def __init__(self, hys=0.25):
        self.hys = float(hys)
        self.prev = {}
        self.n_held = 0
        self.n_total = 0

    def __call__(self, port, value):
        self.n_total += 1
        p = self.prev.get(port)
        if p is None or abs(value - p) >= 0.5 + self.hys:
            p = int(round(value))
        else:
            # Count only the writes the hysteresis actually CHANGED. Plain
            # rounding agrees most of the time (a joint that has not moved a
            # degree yet), and counting those too would inflate this into a
            # number that says nothing about dither.
            self.n_held += int(round(value) != p)
        self.prev[port] = p
        return p

    def reset(self):
        """Forget the held commands (after a park, or a fall recovery)."""
        self.prev.clear()


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

    def __init__(self, n_legs=4, attitude_gain=1.0, hip_offset=None,
                 knee_offset=None):
        super().__init__(n_legs=n_legs)
        # Instance attributes shadow the class ones, so the stance geometry can
        # be raised without editing the module: a knee offset that leaves the
        # legs more extended lifts the trunk, which is the other half of "it is
        # not clearing the ground" when a payload is squashing the servos.
        if hip_offset is not None:
            self.HIP_OFFSET = float(hip_offset)
        if knee_offset is not None:
            self.KNEE_OFFSET = float(knee_offset)
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
    working. It does NOT remove the contact term -- the agreement is automatic,
    so the STOP branch is pinned on; see the module docstring for the resulting
    rescaling of w and gamma.
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


class _TcpEngine:
    """The pyserial-shaped half of the WiFi shim (a port's ``main_engine``).

    ``ardSerial`` only ever touches three things on it -- ``write``, ``readline``
    and ``read_all`` -- so a socket that presents those is indistinguishable from
    a serial port to every layer above.

    Matching pyserial's TIMEOUT SEMANTICS is what actually matters, and it is the
    part a naive ``socket.makefile()`` gets wrong: ``printSerialMessage`` polls
    ``readline()`` in a loop until a line equals the command token, so readline
    must return whatever it has when the timeout expires instead of blocking
    forever, and must not raise on an idle link. A partial line is only ever
    handed back once the deadline has passed, so a response split across TCP
    segments is still reassembled into one line.
    """

    def __init__(self, sock, timeout=WIFI_TIMEOUT):
        self.sock = sock
        self.timeout = float(timeout)
        self.is_open = True
        self._buf = bytearray()

    def _fill(self, timeout):
        """Pull whatever is available within ``timeout``. False = nothing came."""
        if not self.is_open:
            return False
        try:
            self.sock.settimeout(max(0.0, timeout))
            chunk = self.sock.recv(4096)
        except (socket.timeout, BlockingIOError):
            return False
        except OSError:
            self.is_open = False
            return False
        if not chunk:                      # peer closed
            self.is_open = False
            return False
        self._buf.extend(chunk)
        return True

    def _take(self, n):
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def readline(self):
        deadline = time.time() + self.timeout
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                return self._take(nl + 1)
            left = deadline - time.time()
            if left <= 0 or not self.is_open:
                return self._take(len(self._buf))    # partial, as pyserial does
            self._fill(min(left, 0.02))

    def read_all(self):
        while self._fill(0.0):
            pass
        return self._take(len(self._buf))

    def write(self, data):
        self.sock.sendall(data)
        return len(data)

    def close(self):
        self.is_open = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class TcpPort:
    """A ``SerialCommunication.Communication`` stand-in backed by a TCP socket.

    ``PetoiRobot`` keys its ``goodPorts`` dict on port objects and reaches into
    them for exactly four things (``Send_data``, ``Close_Engine`` and, through
    ``main_engine``, ``readline``/``read_all``). Implementing those here means
    the whole stack above -- ``send``, ``sendTask``, the skill compiler, the
    binary 'I' joint packets -- runs over WiFi with no changes at all, and the
    only behavioural difference is latency.
    """

    def __init__(self, host, port=WIFI_PORT, timeout=WIFI_TIMEOUT,
                 connect_timeout=4.0):
        self.port = f"{host}:{port}"       # the 'name' ardSerial prints
        self.bps = 115200
        self.timeout = float(timeout)
        sock = socket.create_connection((host, int(port)), connect_timeout)
        # The bridge sets TCP_NODELAY on its side; without it here, Nagle holds
        # each small command back by up to 40 ms and halves the control rate.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.main_engine = _TcpEngine(sock, timeout)

    def Send_data(self, data):
        self.main_engine.write(data)

    def Open_Engine(self):
        pass                               # created connected

    def Close_Engine(self):
        if self.main_engine is not None:
            self.main_engine.close()


class BittleLink:
    """Link to the robot (a thin wrapper over the vendored Petoi API).

    Three transports, all presenting the same command surface:

    * **USB serial** (default) -- ``PetoiRobot.connectPort`` picks the port.
    * **WiFi** (``wifi_host``) -- a TCP socket to the dongle running
      ``sketch_wifidongle.ino``, injected into ``goodPorts`` as a ``TcpPort`` so
      every layer above is unchanged. Use ``find_bittle.sh`` to get the address.
      Note that Petoi's STOCK dongle sketch will NOT work: it is an HTTP server
      that forwards commands one way and never returns the board's replies, so
      the IMU is unreadable. Re-measure ``--mode rate`` after switching --
      WiFi adds a round trip to every acknowledged command.
    * **dry run** (``dry_run=True``) -- a null transport that accepts every
      command and reports a level, non-falling robot, so the whole experiment
      can be exercised without hardware.
    """

    def __init__(self, dry_run=False, imu_units="auto", roll_sign=1.0,
                 pitch_sign=1.0, keep_gyro=False, wifi_host=None,
                 wifi_port=WIFI_PORT):
        self.dry_run = bool(dry_run)
        self.wifi_host = wifi_host or None
        self.wifi_port = int(wifi_port)
        self.imu_units = imu_units
        self.roll_sign = float(roll_sign)
        self.pitch_sign = float(pitch_sign)
        self.keep_gyro = bool(keep_gyro)
        self._petoi = None
        self._deg = (imu_units != "rad")   # refined by autodetect on first reads
        self._fail_streak = 0
        self.n_imu_fail = 0
        self.n_send_fail = 0
        self.n_imu_stale = 0        # reads identical to the previous one
        self.max_imu_stale = 0      # longest run of them (a freeze, not noise)
        self._stale_streak = 0
        self._last_ypr = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self):
        if self.dry_run:
            print("[link] dry run: no serial connection")
            return
        import PetoiRobot as petoi
        self._petoi = petoi
        if self.wifi_host:
            self._connect_wifi(petoi)
        else:
            petoi.connectPort(petoi.goodPorts)
            if not petoi.goodPorts:
                raise SystemExit("no Bittle found on any serial port -- check the "
                                 "USB adapter and that no other program holds the "
                                 "port")
            print(f"[link] connected: {list(petoi.goodPorts.values())}")
        time.sleep(1.0)
        if not self.keep_gyro:
            self.disable_gyro_balance()
        self.assert_imu_live()

    def _connect_wifi(self, petoi):
        """Open the TCP bridge and confirm a Petoi board is on the other end.

        The identity check is not ceremony: a socket to the wrong thing connects
        just as happily as one to the robot, and every failure downstream would
        then look like a flaky robot instead of a wrong address. '?' is the same
        inert identify token ``PetoiRobot.testPort`` uses over USB -- it moves no
        servos.
        """
        host, port = self.wifi_host, self.wifi_port
        try:
            tcp = TcpPort(host, port)
        except OSError as exc:
            raise SystemExit(
                f"[link] cannot reach the Bittle bridge at {host}:{port} ({exc}).\n"
                f"        Run ./find_bittle.sh to locate the dongle. If it answers "
                f"on port 80 instead, it is running Petoi's STOCK sketch, which "
                f"cannot read the IMU back -- flash sketch_wifidongle.ino.")
        petoi.goodPorts.clear()
        petoi.goodPorts[tcp] = f"{host}:{port}"
        res = petoi.send(petoi.goodPorts, ["?", 0], 3)
        if res == -1:
            petoi.goodPorts.clear()
            tcp.Close_Engine()
            raise SystemExit(
                f"[link] {host}:{port} accepted the connection but no Petoi board "
                f"answered.\n"
                f"        Either the dongle is not bridging to the NyBoard (check "
                f"its serial wiring and that both are powered), or this is the "
                f"stock HTTP sketch rather than sketch_wifidongle.ino.")
        petoi.getModelAndVersion(res)
        print(f"[link] connected over WiFi: {host}:{port}")

    def disable_gyro_balance(self):
        """Turn off the firmware's balancing WITHOUT switching off the IMU.

        The firmware's own balancing fights the CPG for the joints, so it has to
        go -- but on this board the two are separate switches and only one of
        them is safe to touch:

          'gb'  toggles the balancing *behaviour*; the IMU keeps sampling and
                'v' keeps returning fresh angles.  The ack echoes the new state:
                'g' = balancing off, 'G' = balancing on.
          'G'   switches the IMU MODULE off.  'v' still answers -- with the last
                sample it ever took, frozen, forever -- and no amount of 'g'
                brings it back without a reboot.

        ``PetoiRobot.deacGyro()`` sends 'G' on NyBoard-class firmware (measured
        on N_250224), which is why it is not used here: it silently kills the
        attitude feedback, the tilt CUSUM and the fall detector at once, and
        leaves every reading looking plausible.
        """
        if self.dry_run:
            return
        state = "?"
        for _ in range(2):
            res = self._send(["gb", 0], timeout=1)
            state = str(res[0])[0] if isinstance(res, (list, tuple)) and res else "?"
            if state == "g":                       # balancing now off
                print("[link] gyro balancing off (IMU still sampling)")
                return
            if state != "G":                       # unrecognised ack -- do not guess
                break
        print("[link] !! could not confirm the gyro-balance state (ack "
              f"{state!r}); the firmware may fight the CPG for the joints")

    def assert_imu_live(self, n=8, dt=0.08):
        """Fail loudly if the IMU answers but never changes.

        A frozen IMU is the one hardware fault that looks like success: every
        read parses, every value is plausible, and the attitude feedback simply
        does nothing.  Even a robot sitting still on a stand jitters in the last
        decimal, so bit-identical samples over ~1 s mean the module is off.
        """
        if self.dry_run:
            return True
        vals = []
        for _ in range(int(n)):
            imu = self.read_imu()
            if imu is not None:
                vals.append(tuple(imu[:3]))
            time.sleep(dt)
        if len(vals) < max(2, n // 4):
            raise SystemExit(
                "[link] the IMU returned almost nothing. Some firmware silences "
                "it when gyro balancing is off -- retry with --keep-gyro.")
        if len(set(vals)) == 1:
            raise SystemExit(
                "[link] the IMU is FROZEN: {} reads returned the identical "
                "sample {}.\n"
                "        The IMU module is switched off -- 'v' keeps answering "
                "with the last sample it took, so roll and pitch look like a "
                "robot that never moves.\n"
                "        Power-cycle the robot (the switch does not come back "
                "without a reboot) and try again; if it persists, run with "
                "--keep-gyro, which leaves the firmware's balancing on."
                .format(len(vals), tuple(round(float(v), 5) for v in vals[0])))
        print(f"[link] IMU live ({len(set(vals))}/{len(vals)} distinct samples)")
        return True

    def close(self):
        if self.dry_run or self._petoi is None:
            return
        try:
            self.posture("rest", delay=1.0)
        finally:
            self._petoi.closeAllSerial(self._petoi.goodPorts)
            print(f"[link] {'wifi' if self.wifi_host else 'serial'} closed")

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
        """Command joints as a flat [index, deg, index, deg, ...] list.

        The 'I' token is the SIMULTANEOUS binary move -- one packet sets every
        listed joint at once (``ardSerial``'s skill compiler builds a single row
        for 'i'/'I', and one row per joint for 'm'), 18 bytes, 1.25 ms on the
        wire at 115200. Everything else in the ~35 ms tick is the firmware.

        Not waiting for the token echo does NOT buy rate, measured on N_250224:
        the board will swallow joint packets at ~40 Hz when nothing else is
        asked of it, but a 'v' sent straight after a binary packet is consumed
        as payload (11/40 IMU reads survived), and the 5 ms settle that fixes
        that leaves the tick at 36.1 ms -- slower than the 34.8 ms acknowledged
        path, because the IMU read then costs ~31 ms instead of ~7 ms. The
        firmware's per-command work is conserved either way. ``--imu-every``
        (which skips whole reads) is the only lever that actually moves the rate.
        """
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
        # A freeze mid-run (a firmware hiccup, a stray 'G') is silent otherwise:
        # the values still parse, they just stop moving.
        if (yaw, pitch, roll) == self._last_ypr:
            self._stale_streak += 1
            self.n_imu_stale += 1
            self.max_imu_stale = max(self.max_imu_stale, self._stale_streak)
        else:
            self._stale_streak = 0
        self._last_ypr = (yaw, pitch, roll)
        if self.imu_units == "auto" and max(abs(yaw), abs(pitch), abs(roll)) > 6.5:
            self._deg = True               # radians never exceed 2*pi
        if self._deg:
            yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
        return (self.roll_sign * roll, self.pitch_sign * pitch, yaw,
                tuple(vals[3:6]))

    # ── bring-up helpers ─────────────────────────────────────────────────────
    def measure_period(self, n=60, pose=None, read_imu=True):
        """Median and worst wall-clock cost of one control tick.

        With ``read_imu`` the tick is a joint write PLUS an IMU read; without, it
        is the write alone. The two differ by several milliseconds on a Bittle,
        which is what makes ``--imu-every`` worth anything -- see
        ``amortised_period``. Measured on the robot, this cost is the firmware's
        turnaround, not the host's: removing the vendored sleeps in
        ``PetoiRobot.ardSerial`` changes it by less than a millisecond.
        """
        pose = pose if pose is not None else self.neutral_pose()
        dts = []
        for _ in range(int(n)):
            t0 = time.perf_counter()
            self.joints(pose)
            if read_imu:
                self.read_imu()
            dts.append(time.perf_counter() - t0)
        return float(np.median(dts)), float(np.max(dts))

    def amortised_period(self, imu_every=1, n=60, pose=None):
        """Cost of an average control tick when the IMU is read every k-th one.

        Every tick pays the joint write; only one in ``imu_every`` also pays the
        IMU read. Multiplying the combined cost by ``imu_every`` (as this used
        to) charges every tick for a read it does not do, and makes --imu-every
        pick a dt several times too large -- i.e. it made the control rate WORSE
        the more you amortised.
        """
        t_full, worst_full = self.measure_period(n=n, pose=pose, read_imu=True)
        if imu_every <= 1:
            return t_full, worst_full, t_full, t_full
        t_write, worst_write = self.measure_period(n=n, pose=pose, read_imu=False)
        extra = max(0.0, t_full - t_write)
        return (t_write + extra / float(imu_every), max(worst_full, worst_write),
                t_write, t_full)

    @staticmethod
    def neutral_pose(shift_deg=None, hip_deg=None, knee_deg=None,
                     shift_port=None):
        """Standing pose in the indexed [port, deg, ...] form.

        ``hip_deg``/``knee_deg`` override the offsets, so a run that raises the
        stance parks at the SAME geometry it walks at -- otherwise the robot
        drops to the default crouch every time it settles or recovers.

        ``shift_port`` MUST be passed by any caller that honours ``--shift-port``.
        Falling back to the module constant parks a different servo than the one
        the control loop drives: the harness is then never recentred, and port
        ``SHIFT_PORT`` gets driven to the harness angle instead.
        """
        hip = HIP_OFFSET_DEG if hip_deg is None else float(hip_deg)
        knee = KNEE_OFFSET_DEG if knee_deg is None else float(knee_deg)
        out = []
        for j in range(4):
            out += [HIP_PORTS[j], int(round(hip)),
                    KNEE_PORTS[j], int(round(knee))]
        if shift_deg is not None:
            out += [SHIFT_PORT if shift_port is None else int(shift_port),
                    int(round(shift_deg))]
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
                 att_tau=0.06, cmd_hys=0.25, hip_offset=None, knee_offset=None,
                 recover="manual", recover_skill="up", recover_pause=5.0,
                 settle_t=1.5, cpg_dt=0.01, synthetic=None):
        self.link = link
        self.dt = float(dt)          # control period (serial-limited)
        self.contacts = contacts
        self.cpg_dt = float(cpg_dt)  # oscillator integration step (see control_tick)
        self.shift_port = int(shift_port)
        # Every control tick packs the eight leg angles and the shift angle into
        # one 'I' write. If the harness servo sits on a leg pin the two commands
        # collide inside that packet -- the gait would silently lose a joint, so
        # refuse rather than walk a three-legged robot.
        if self.shift_port in set(HIP_PORTS) | set(KNEE_PORTS):
            raise ValueError(
                f"shift_port={self.shift_port} is a LEG joint "
                f"(hips {HIP_PORTS}, knees {KNEE_PORTS}). Move the mass-shifter "
                f"servo to a spare NyBoard pin (1-7) and pass --shift-port.")
        self.shift_centered = float(shift_centered)
        self.shift_shifted = float(shift_shifted)
        self.manual_shift = bool(manual_shift)
        self.imu_every = max(1, int(imu_every))
        self.attitude = bool(attitude)
        self.attitude_gain = float(attitude_gain)
        # Smoothing (identical to stand_test.py, so the bench and the experiment
        # run the same law): the raw attitude is noisy at walking vibration, and
        # feeding it straight to the knee correction buzzes the servos.
        self.hip_offset = hip_offset       # None = the module defaults
        self.knee_offset = knee_offset
        self.att_tau = float(att_tau)
        self.quant = CommandQuantizer(cmd_hys)
        self._att_f = None                 # filtered (roll, pitch), radians
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
        # What the harness servo was ACTUALLY commanded this bout. A wrong
        # --shift-centered/--shift-shifted pair cannot be detected from inside
        # the loop -- nothing on the link reports a stall, so a rack driven past
        # its end stop just buzzes against the guard -- so the commanded range is
        # reported at the end of the bout to be compared against --find-travel.
        self.shift_cmd_min = self.shift_cmd_max = None
        self.n_shift_cmd = 0
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
        roll_f, pitch_f = self._filter_attitude(roll, pitch)
        att = (roll_f, pitch_f) if self.attitude else (None, None)
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
            cmd += [HIP_PORTS[j], self.quant(HIP_PORTS[j], h),
                    KNEE_PORTS[j], self.quant(KNEE_PORTS[j], k)]
            self.last_cmd[2 * j] = np.deg2rad(h)
            self.last_cmd[2 * j + 1] = np.deg2rad(k)
        if self.shift_port >= 0 and not self.manual_shift:
            shift_deg = self.quant(self.shift_port, self._shift_deg(frac))
            cmd += [self.shift_port, shift_deg]
            self.n_shift_cmd += 1
            d = float(shift_deg)
            self.shift_cmd_min = d if self.shift_cmd_min is None else min(self.shift_cmd_min, d)
            self.shift_cmd_max = d if self.shift_cmd_max is None else max(self.shift_cmd_max, d)
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

    def _filter_attitude(self, roll, pitch):
        """Low-pass the attitude before it reaches the knee correction.

        Walking vibration puts several degrees of noise on the Petoi IMU, and the
        correction it drives is only a few degrees wide (``DKNEE_CLIP``), so the
        unfiltered signal spends the whole budget chasing noise. The filter runs
        on the CONTROL period, which is the real sampling period -- the raw
        values are still what the fall check and the logs see.
        """
        if self.att_tau <= 0.0:
            return roll, pitch
        a = lpf_alpha(self.att_tau, self.dt)
        if self._att_f is None:
            self._att_f = (roll, pitch)
        else:
            self._att_f = (self._att_f[0] + a * (roll - self._att_f[0]),
                           self._att_f[1] + a * (pitch - self._att_f[1]))
        return self._att_f

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
        # Both firmware skills fired above ('k<recover_skill>' and 'kbalance')
        # switch the firmware's balancing back ON, undoing the 'gb' that
        # connect() sent. Left on, the firmware and this control loop both write
        # servo targets every tick: the legs fight the CPG, and the harness
        # servo -- on a port the firmware also drives -- hammers between the two
        # targets for the rest of the bout. Nothing else re-asserts it, so one
        # fall would otherwise poison every event after it. disable_gyro_balance
        # is ack-checked, so re-sending the toggle is safe.
        if not self.link.keep_gyro:
            self.link.disable_gyro_balance()
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
        cpg = BittleCPG(n_legs=4, attitude_gain=self.attitude_gain,
                        hip_offset=self.hip_offset, knee_offset=self.knee_offset)
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
            shift_deg=self.shift_centered if self.shift_port >= 0 else None,
            hip_deg=self.hip_offset, knee_deg=self.knee_offset,
            shift_port=self.shift_port))
        # The park bypasses the quantizer and the robot is about to be handled,
        # so both smoothers are holding stale state: start the next tick clean.
        self.quant.reset()
        self._att_f = None
        time.sleep(0.5)
