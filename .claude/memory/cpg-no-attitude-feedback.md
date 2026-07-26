---
name: cpg-no-attitude-feedback
description: our Hopf CPG has Righetti contact feedback but NO body-attitude feedback — root cause of frequent falls; SOTA adds VMC/IMU feedback or CPG-RL
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f8f0747-1c9d-4007-96d6-8da8b667cf75
---

The Hopf-oscillator CPG (`JointCPG.step` in methods/marxefe_optimizer.py:669) takes only
`(params_8d, raw_contacts, dt)`. Its STOP_GAIN/F_FAST feedback is the Righetti & Ijspeert
(ICRA 2008) touch-sensor feedback: it corrects swing/stance **gait timing / stumbling**, NOT
body balance. Base orientation (roll/pitch) never enters cpg.step; joint commands are
open-loop POSITION_CONTROL targets. So trunk tilt accumulates unchecked on slopes/low-friction
bands until the robot topples.

**Why it matters:** the high fall rate is largely a CONTROLLER-STRUCTURE artifact, not terrain
difficulty. Explains [[cpg-terrain-feasibility-envelope]] (open-loop viable only flat + ~10°)
and why MARX-EFE can at best match no-adapt ([[track1-marxefe-no-adaptation]]): parameter-level
adaptation is a slow, indirect substitute for the fast posture loop the controller lacks. A
reviewer can call the baseline a straw man (missing the attitude-feedback layer standard in the
CPG literature it cites).

**SOTA (lit search 2026-07):** Hopf/coupled-oscillator CPG is a sound classic but NOT SOTA alone.
Two standard fixes, both target this failure mode: (1) CPG + body-attitude (IMU) feedback via
Virtual Model Control — "body attitude adopted as sensory feedback... modulated to walk on slope"
(the [[bo-flat-falls-diagnosis]] memory hints VMC existed in an earlier version); (2) CPG-RL
(Bellegarda & Ijspeert 2022, arXiv:2211.00458) learns to modulate oscillator amp/freq, robust to
115%-mass load, sim-to-real A1. Pure DRL policies now dominant for rough terrain (Ha et al. 2025,
arXiv:2406.01152).

**IMPLEMENTED 2026-07 (validated):** added VMC-style attitude feedback to JointCPG.step (now takes
optional roll/pitch; open-loop if omitted, so backward-compatible). Virtual PD on trunk roll/pitch
distributed to knees (leg-length): roll on raw angle (KP_ROLL=0.8), pitch on deviation from a slow
EMA baseline (KP_PITCH=0.5, ATT_EMA_ALPHA=0.01) so a steady incline is tolerated; DKNEE_CLIP=0.35.
Per-leg signs _FRONT=[1,1,-1,-1], _LEFT=[1,-1,1,-1] (legs 0=FL,1=FR,2=RL,3=RR). Empirically-validated
signs ROLL_SIGN=PITCH_SIGN=-1 (sweep of ±/± on 15° slope). Class flag ATTITUDE_FEEDBACK (default True).
Wired into experiment-flat2slope-adapt AND experiment-natural-adapt run loops (causal: previous-step
roll/pitch), each with a `--no-attitude-fb` CLI ablation (env CPG_ATTITUDE_FB). NOT wired into
run_bo_online / run_episode_maxrefe (offline optima still open-loop).

**Validation (flat-optimal gait, standalone sweep 5 seeds):** flat 0/5 both (tilt 2.4->2.0, no cost);
10° 0/5 both (13.2->11.3); **15° falls 3/5->0/5, max tilt 55°->17°** (envelope 10°->15°); 20° fails at
spawn both (outside envelope, separate issue). Real flat2slope harness, noadapt slope15, 4 seeds:
falls 4/4 (OFF, down by 3-6s) -> 2/4 (ON, survivors reach full 20s, meanJ up to 0.60).

**Next:** offline optima theta*(flat)/theta*(10°) were computed OPEN-LOOP, so with feedback ON they
are slightly off-optimal incumbents; re-derive with feedback ON for a clean paper. Gains are a first
cut (could tune). Then re-run flat->slope ON-vs-OFF + natural ON-vs-OFF for the paper comparison.
