---
name: bo-flat-falls-diagnosis
description: "Why BO on flat terrain fell a lot after switching to the paper's CPG parameterization; the controller-mapping regression"
metadata: 
  node_type: memory
  type: project
  originSessionId: be67e37b-468a-4e39-80e6-d80e526d524b
---

Investigated (2026-07-01) why BO on flat terrain (`experiment-flat`) produced ~57% falls
and couldn't match Zhang et al. IROS 2024 ("Online Optimization of CPGs...", in `literature/`).

Root causes, in order of impact:
1. **Controller regression (the real blocker).** Commit e7cb08e replaced a *working*
   direct-joint-angle CPG (`hip_angle = offset + amp·osc`, `knee_angle = offset − amp·max(0,·)`;
   ~7% falls, J≈2.4) with the paper's Cartesian foot-trajectory (eqs 4–5) + empirical Laikago
   IK (`leg_ik` in `methods/cpg_bounds.py`). The Cartesian+IK mapping under pure POSITION_CONTROL
   does NOT walk: with hand-set mid-range params it rolls over (roll RMS 30–65°) and tips in
   1–3 s. Best partial gait: `H_LEG=0.33, mu=1.3` moved 1.85 m @ 0.72 m/s but still tipped at ~3.2 s.
   Larger H_LEG overextends and tips faster.
2. **Objective was saturated/broken.** Old objective rewarded absolute forward POSITION reaching
   4.0 m with σ=1.0 m, but the robot only reaches ~1 m → reward ≈0 everywhere reachable → BO's
   optimum was to stand still (best trial had all params pinned to lower bounds, J≈−0.01). FIXED:
   rewrote `compute_objective` to velocity tracking (paper eq 9, v*=0.5 m/s, width 0.05, cap 0.85,
   −0.5·CoT). Now standing still scores ~0.
3. **Control rate 10× too coarse.** Was dt=0.01 (100 Hz); paper is 1 kHz. FIXED: `CONTROL_DT=0.002`
   (500 Hz) module constant in `methods/bo_optimizer.py`.
4. **No VMC.** Added an attitude-PD foot-height posture controller (`USE_VMC`) but it is
   INSUFFICIENT for this Cartesian controller — cm-scale foot correction can't damp 30–65° roll;
   made falls *sooner* for every gain/sign. Left OFF by default. Faithful fix = torque-based VMC
   (Jacobian-transpose trunk torques) which needs TORQUE_CONTROL.

RESOLUTION (user chose: revert to joint-angle controller + keep objective; port objective to MARX-EFE):
- `methods/bo_optimizer.py`: run_cpg_trial reverted to the joint-space Righetti CPG (spliced from HEAD),
  velocity objective kept, VMC removed. **CONTROL_DT reverted to 0.01 (100 Hz)** — the joint-space
  controller is rate-dependent (step-based debounce, F_FAST/STOP) and tips at 200/500 Hz; walks cleanly
  at 100 Hz. (The rate bump only mattered for the dropped Cartesian controller.)
- `methods/cpg_bounds.py`: `bounds` reverted to joint-space 8D; Cartesian constants kept only for the
  exploratory (non-comparison) `gpefe_optimizer`.
- `methods/marxefe_optimizer.py`: added self-contained `JointCPG` class; both control loops
  (`run_episode_maxrefe`, `run_bo_online`) + `reset_simulation` route through it; `compute_objective`
  ported to the identical velocity objective; agent goal AND objective both use `target_velocity`.
- `experiment-flat/run_multiseed.py`: BO/grid/MARX-EFE all use `PAPER_TARGET_VELOCITY = 0.5`.

VALIDATED: BO 12-trial scratch run dropped falls 57%→17% and BO tracks v* (walked at 0.50 m/s).
Full grid/BO/MARX-EFE pipeline runs end-to-end. NOTE: legacy CSV columns are still named
gc/gp/omegaswing/… but now hold the joint-space params [coupling_gain,w_swing,…] (cosmetic).
CPG param order is [coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN, hip_amp, knee_amp, b].
