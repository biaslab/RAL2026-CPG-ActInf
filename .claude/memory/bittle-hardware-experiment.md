---
name: bittle-hardware-experiment
description: 2026-07-27 experiment-real/ runs the payload-shift experiment on the Bittle through the SAME methods/ driver; CPG must be sub-stepped because explicit Euler diverges at the serial control rate
metadata:
  type: project
---

Built 2026-07-27: `experiment-real/run_experiment.py` + `bittle_interface.py` run the
continual payload-shift bout on the real Bittle through
`methods.continual_driver.run_event_bout` — the same detector, arms and CSV schema as
`experiment-simulation/experiment-payload-adapt/`. `BittlePhysics` implements the
physics contract; `BittleCPG` subclasses `JointCPG`. To make that sharing possible,
`JointCPG`/`PerLegCPG` were extracted from `methods/marxefe_optimizer.py` (which
imports pybullet at module level) into **`methods/cpg_controller.py`**, re-exported so
every existing `from methods.marxefe_optimizer import JointCPG` still works.

**The load-bearing discovery:** the CPG's explicit-Euler discretization is only stable
at the simulator's 100 Hz. Measured over 200 random gaits from the search box:
0/200 diverge at dt=0.01, **103/200 at dt=0.025, 191/200 at dt=0.03** — i.e. at the
40 Hz a Petoi serial link actually sustains, half the search space produces NaN joint
targets. `BittleCPG.control_tick()` therefore sub-steps the oscillators at `--cpg-dt`
(0.01) inside each control tick and commands only the last sub-step; the VMC attitude
correction is applied once per control tick, since that (not the integration step) is
the real IMU sampling period. This is why `JointCPG.attitude_dknee()` was split out of
`step()`.

Other hardware facts worth not rediscovering:
- Petoi leg indices run clockwise from front-left, so `HIP_PORTS = [8, 9, 11, 10]`
  maps the simulator's `[FL, FR, RL, RR]`; swapping the rear pair silently breaks the
  trot's diagonal pairing.
- The degrees mapping (hip 40 deg offset / 120 deg per unit, knee 30 / 12) is anchored
  so the simulation incumbent reproduces the hand-tuned gait in `petoi_Hopf.py`; their
  w_swing/w_stance already agree (13.0/25.0 vs 12/24).
- No foot-contact sensors and no odometry. Contacts default to what the oscillator
  phase expects; `--vx-source none` reports a constant speed, which zeroes the
  detector's speed-deficit term and leaves a **tilt-only** CUSUM. Logged distances are
  dead-reckoned, not measured — see [[backward-gait-ceiling]] for why falls are the
  only honest metric anyway.
- `--dry-run` rehearses the whole chain with a null serial transport plus a caricature
  `SyntheticRobot`; it is for plumbing only, never for numbers.
- **Attitude convention: `pitch > 0` is nose UP, `roll > 0` is right-side DOWN.**
  `get_observation` unpacks the simulated euler angles as `pitch, roll, yaw` with the
  robot walking in +Y, so euler[0] (about world X) is a nose-up-positive pitch and
  euler[1] (about the forward axis) is a right-down-positive roll — and
  `JointCPG.ROLL_SIGN/PITCH_SIGN = -1,-1` were fit against THAT. `--mode imu` and the
  README told the operator the opposite for pitch until 2026-07-28; following the old
  text sets `--pitch-sign -1` and makes the pitch channel drive the robot over.
- A regression of the knee correction on roll/pitch can NEVER catch an inverted IMU:
  the correction is computed from whatever attitude it was handed, so the IMU sign
  cancels. Only a physical two-attitude check against a level reference
  (`stand_test.py --mode sign`) or the operator's eyes establish it.
- `experiment-real/stand_test.py` is the on-a-stand bench test of the control loop
  (`--mode sign|still|walk`, `--inject` for a synthetic attitude, `--no-attitude` for
  the open-loop reference): same `control_tick` + VMC path as a bout, with the harness,
  detector, responders and fall logic removed. It reports achieved rate, joint travel
  vs the safety limits, clip saturation, and how often the correction fell below the
  servos' **1 deg quantum** — which is a quarter of the 4.2 deg `DKNEE_CLIP`, so
  posture authority on hardware is coarse.

**Why:** anyone touching the hardware path will hit the integration-rate trap first,
and it manifests as a mid-run crash (or slammed servos), not as an obviously wrong
number.

**How to apply:** keep `--cpg-dt` at the simulation step whenever `--dt` is raised;
if the incumbent gait is retuned on the robot, save it to
`experiment-real/results/incumbent.json` rather than editing the constant. Related:
[[consolidated-continual-experiments]], [[payload-shift-experiment]],
[[repo-layout-2026-07-27]].
