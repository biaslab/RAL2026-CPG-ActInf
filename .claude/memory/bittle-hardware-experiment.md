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

**Why:** anyone touching the hardware path will hit the integration-rate trap first,
and it manifests as a mid-run crash (or slammed servos), not as an obviously wrong
number.

**How to apply:** keep `--cpg-dt` at the simulation step whenever `--dt` is raised;
if the incumbent gait is retuned on the robot, save it to
`experiment-real/results/incumbent.json` rather than editing the constant. Related:
[[consolidated-continual-experiments]], [[payload-shift-experiment]],
[[repo-layout-2026-07-27]].
