"""Joint-space Righetti-style CPG controllers (pure NumPy, no simulator).

Extracted from ``methods.marxefe_optimizer`` so the SAME controller can be driven
by the PyBullet experiments and by the real Bittle (``experiment-real/``) without
dragging in pybullet/torch. ``marxefe_optimizer`` re-exports both classes, so
``from methods.marxefe_optimizer import JointCPG`` keeps working unchanged.

  * :class:`JointCPG`  -- 8-D global gait vector (methods.cpg_bounds order).
  * :class:`PerLegCPG` -- 11-D variant with a per-leg hip amplitude (leg damage).

The hardware mapping to Bittle joint angles lives in
``experiment-real/bittle_interface.py``; it subclasses :class:`JointCPG` and only
overrides the CPG-state -> joint-angle conversion, so the oscillator dynamics,
contact feedback and attitude feedback are literally the same code in sim and on
the robot.
"""

import numpy as np


class JointCPG:
    """Self-contained joint-space Righetti-style CPG (the validated controller).

    Encapsulates the four coupled Hopf oscillators, the Righetti STOP/FAST
    contact feedback, the per-leg phase hysteresis, and the CPG→joint mapping.
    One instance per episode; call :meth:`step` each control tick.

    8-D parameter vector (matches ``methods.cpg_bounds``):
      [coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN, hip_amp, knee_amp, b]

    This is the joint-space mapping shared with ``methods.bo_optimizer`` — it
    replaces the Zhang-et-al. Cartesian foot-trajectory + IK controller, which
    was laterally unstable on Laikago under position control.
    """

    ALPHA = 3.0
    BETA = 12.0
    U = 2.0
    HIP_OFFSET = 0.26
    KNEE_OFFSET = -1.0
    SWING_ENTER, SWING_EXIT = 0.15, 0.02
    STANCE_ENTER, STANCE_EXIT = -0.15, -0.02
    DEBOUNCE_THRESHOLD = 2
    K = np.array([[0, -1, -1, 1],
                  [-1, 0, 1, -1],
                  [-1, 1, 0, -1],
                  [1, -1, -1, 0]], dtype=float)

    # ── VMC-style body-attitude feedback (posture leveling) ──────────────────
    # A virtual PD on the trunk roll/pitch, distributed to the legs as knee
    # (leg-length) corrections: a leg on the sinking side straightens to push
    # that corner back up, so the trunk stays level. Inactive unless roll/pitch
    # are supplied to step(), so the open-loop controller is the default and all
    # existing call sites are unchanged. Roll acts on the raw angle (straight
    # walking has no sustained bank, and lateral tip-over is the main hazard);
    # pitch acts on the deviation from a slow baseline so a steady incline offset
    # is tolerated rather than fought. Signs are validated empirically.
    ATTITUDE_FEEDBACK = True
    KP_ROLL, KD_ROLL = 0.8, 0.05        # knee rad per rad / per rad/s of roll
    KP_PITCH, KD_PITCH = 0.5, 0.05      # knee rad per rad / per rad/s of pitch
    ROLL_SIGN, PITCH_SIGN = -1.0, -1.0  # set by empirical sign check (slope test)
    ATT_EMA_ALPHA = 0.01                # pitch baseline leak (~1 s at 100 Hz)
    DKNEE_CLIP = 0.35                   # max knee correction [rad]
    # per-leg geometry signs for legs [0=FL, 1=FR, 2=RL, 3=RR]
    _FRONT = np.array([1.0, 1.0, -1.0, -1.0])   # +front / -hind (pitch)
    _LEFT = np.array([1.0, -1.0, 1.0, -1.0])    # +left / -right (roll)

    def __init__(self, n_legs=4):
        self.n = n_legs
        theta = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        self.x = np.array([np.sqrt(self.U) * np.cos(t) for t in theta])
        self.y = np.array([np.sqrt(self.U) * np.sin(t) for t in theta])
        self.deb = np.zeros(n_legs, dtype=int)
        self.cc = np.zeros(n_legs, dtype=int)
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.pitch_ema = 0.0
        self._att_init = False
        # attitude-feedback gains as instance state, so they can be adapted online
        # (default to the validated class constants); order [kp_roll, kd_roll,
        # kp_pitch, kd_pitch]. Set via set_gains().
        self.kp_roll, self.kd_roll = self.KP_ROLL, self.KD_ROLL
        self.kp_pitch, self.kd_pitch = self.KP_PITCH, self.KD_PITCH
        self.phase = []
        for j in range(n_legs):
            if self.y[j] > self.SWING_ENTER:
                self.phase.append("swing")
            elif self.y[j] < self.STANCE_ENTER:
                self.phase.append("stance")
            else:
                self.phase.append("transition")

    def _get_phase(self, y_val, j):
        s = self.phase[j]
        if s == "swing":
            s = "transition" if y_val < self.SWING_EXIT else "swing"
        elif s == "stance":
            s = "transition" if y_val > self.STANCE_EXIT else "stance"
        else:
            if y_val > self.SWING_ENTER:
                s = "swing"
            elif y_val < self.STANCE_ENTER:
                s = "stance"
            else:
                s = "transition"
        self.phase[j] = s
        return s

    def step(self, params_8d, raw_contacts, dt, roll=None, pitch=None):
        """Advance one control tick; return (hip_angles, knee_angles) arrays.

        If the trunk ``roll`` and ``pitch`` (rad) are supplied, a VMC-style
        posture-leveling term is added to the knee targets (see the class-level
        attitude-feedback constants); if omitted, the controller is open-loop."""
        coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN, hip_amp, knee_amp, b = params_8d
        x_prev, y_prev = self.x.copy(), self.y.copy()

        # radial state + intrinsic frequency (swing/stance blended by y)
        w_vec = w_stance / (np.exp(-b * y_prev) + 1.0) + w_swing / (np.exp(b * y_prev) + 1.0)
        r_vec = np.sqrt(x_prev ** 2 + y_prev ** 2)
        x_new = x_prev + dt * (self.ALPHA * (self.U - r_vec ** 2) * x_prev - w_vec * y_prev)

        # contact debounce
        raw = np.asarray(raw_contacts, dtype=int)
        for j in range(self.n):
            if raw[j] == self.deb[j]:
                self.cc[j] = 0
            else:
                self.cc[j] += 1
            if self.cc[j] >= self.DEBOUNCE_THRESHOLD:
                self.deb[j] = raw[j]
                self.cc[j] = 0

        # STOP/FAST feedback per leg
        phases = [self._get_phase(y_prev[j], j) for j in range(self.n)]
        coupling_y = coupling_gain * (self.K @ y_prev)
        u_fb = np.zeros(self.n)
        for j in range(self.n):
            in_stop = ((phases[j] == "swing" and self.deb[j] < 0.5) or
                       (phases[j] == "stance" and self.deb[j] > 0.5))
            if in_stop:
                u_fb[j] = STOP_GAIN * (w_vec[j] * x_prev[j] - coupling_y[j])
            elif phases[j] in ("swing", "stance"):
                u_fb[j] = np.sign(y_prev[j]) * F_FAST

        y_new = y_prev + dt * (self.BETA * (self.U - r_vec ** 2) * y_prev
                               + w_vec * x_prev + coupling_y + u_fb)

        self.x, self.y = x_new, y_new
        hip_angles = self.HIP_OFFSET + hip_amp * x_new
        knee_angles = self.KNEE_OFFSET - knee_amp * np.maximum(0.0, y_new)

        # VMC-style attitude feedback: level the trunk by adjusting leg length.
        if self.ATTITUDE_FEEDBACK and roll is not None and pitch is not None:
            knee_angles = knee_angles + self.attitude_dknee(roll, pitch, dt)
        return hip_angles, knee_angles

    def attitude_dknee(self, roll, pitch, dt):
        """Per-leg knee correction of the VMC posture-leveling term.

        Split out of :meth:`step` because ``dt`` here is the ATTITUDE SAMPLING
        period (it only sets the roll/pitch rate estimate), which on hardware is
        the control period and not the oscillator integration step -- those two
        coincide in simulation but not when :meth:`step` is sub-stepped (see
        ``experiment-real/bittle_interface.py``)."""
        if not self._att_init:                           # seed baselines/derivs
            self.prev_roll, self.prev_pitch = roll, pitch
            self.pitch_ema = pitch
            self._att_init = True
        roll_rate = (roll - self.prev_roll) / dt
        pitch_rate = (pitch - self.prev_pitch) / dt
        self.prev_roll, self.prev_pitch = roll, pitch
        self.pitch_ema += self.ATT_EMA_ALPHA * (pitch - self.pitch_ema)
        d_pitch = pitch - self.pitch_ema                 # tolerate steady incline
        roll_cmd = self.ROLL_SIGN * (self.kp_roll * roll + self.kd_roll * roll_rate)
        pitch_cmd = self.PITCH_SIGN * (self.kp_pitch * d_pitch + self.kd_pitch * pitch_rate)
        return np.clip(self._FRONT * pitch_cmd + self._LEFT * roll_cmd,
                       -self.DKNEE_CLIP, self.DKNEE_CLIP)

    def set_gains(self, gains):
        """Set the 4 attitude-feedback gains [kp_roll, kd_roll, kp_pitch, kd_pitch].
        The online-adaptable channel (continuous, no gait-phase discontinuity)."""
        self.kp_roll, self.kd_roll, self.kp_pitch, self.kd_pitch = \
            (float(g) for g in gains)


class PerLegCPG(JointCPG):
    """JointCPG variant with a PER-LEG hip amplitude, for the leg-damage
    experiment. Identical oscillators / coupling / contact feedback / attitude
    feedback -- the only change is that the single global ``hip_amp`` becomes a
    4-vector (one per leg), so an asymmetric fault (a weak leg) can be compensated
    by lowering that leg's swing amplitude (less torque demanded, less drag) and
    leaning on the others -- a compensation the global-symmetric gait cannot
    express. The global incumbent is this controller with all four hip amplitudes
    equal, so no-adapt still fails under damage.

    11-D parameter vector:
      [coupling, w_swing, w_stance, F_FAST, STOP,
       hipA_FL, hipA_FR, hipA_RL, hipA_RR, kneeA, b]
    """

    def step(self, params, raw_contacts, dt, roll=None, pitch=None):
        coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN = params[:5]
        hip_amp = np.asarray(params[5:9], float)      # per-leg (4,)
        knee_amp, b = float(params[9]), float(params[10])
        x_prev, y_prev = self.x.copy(), self.y.copy()

        w_vec = w_stance / (np.exp(-b * y_prev) + 1.0) + w_swing / (np.exp(b * y_prev) + 1.0)
        r_vec = np.sqrt(x_prev ** 2 + y_prev ** 2)
        x_new = x_prev + dt * (self.ALPHA * (self.U - r_vec ** 2) * x_prev - w_vec * y_prev)

        raw = np.asarray(raw_contacts, dtype=int)
        for j in range(self.n):
            if raw[j] == self.deb[j]:
                self.cc[j] = 0
            else:
                self.cc[j] += 1
            if self.cc[j] >= self.DEBOUNCE_THRESHOLD:
                self.deb[j] = raw[j]
                self.cc[j] = 0

        phases = [self._get_phase(y_prev[j], j) for j in range(self.n)]
        coupling_y = coupling_gain * (self.K @ y_prev)
        u_fb = np.zeros(self.n)
        for j in range(self.n):
            in_stop = ((phases[j] == "swing" and self.deb[j] < 0.5) or
                       (phases[j] == "stance" and self.deb[j] > 0.5))
            if in_stop:
                u_fb[j] = STOP_GAIN * (w_vec[j] * x_prev[j] - coupling_y[j])
            elif phases[j] in ("swing", "stance"):
                u_fb[j] = np.sign(y_prev[j]) * F_FAST

        y_new = y_prev + dt * (self.BETA * (self.U - r_vec ** 2) * y_prev
                               + w_vec * x_prev + coupling_y + u_fb)
        self.x, self.y = x_new, y_new
        hip_angles = self.HIP_OFFSET + hip_amp * x_new          # per-leg amplitude
        knee_angles = self.KNEE_OFFSET - knee_amp * np.maximum(0.0, y_new)

        if self.ATTITUDE_FEEDBACK and roll is not None and pitch is not None:
            knee_angles = knee_angles + self.attitude_dknee(roll, pitch, dt)
        return hip_angles, knee_angles

    @staticmethod
    def expand8(p8):
        """8-D global gait -> 11-D per-leg (hip amplitude replicated across legs)."""
        p8 = np.asarray(p8, float)
        return np.concatenate([p8[:5], [p8[5]] * 4, [p8[6]], [p8[7]]])

    @staticmethod
    def expand_box(lo8, hi8):
        """8-D box -> 11-D box (hip-amplitude bounds replicated across legs)."""
        lo8, hi8 = np.asarray(lo8, float), np.asarray(hi8, float)
        lo = np.concatenate([lo8[:5], [lo8[5]] * 4, [lo8[6]], [lo8[7]]])
        hi = np.concatenate([hi8[:5], [hi8[5]] * 4, [hi8[6]], [hi8[7]]])
        return lo, hi
