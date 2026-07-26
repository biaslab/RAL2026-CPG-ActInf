---
name: cusum-suboptimality-trigger
description: CUSUM re-thresholding of the DT Newton-decrement trigger — fires early on sustained recoverable transitions (gravel) not at the fall; silent on un-recoverable ice
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f8f0747-1c9d-4007-96d6-8da8b667cf75
---

Fixes the DT-fires-too-late problem ([[attitude-feedback-subsumes-adaptation]],
[[lambda2-trigger-fails-detection]]) WITHOUT abandoning principled suboptimality detection.

**Diagnosis (seed 0 hard terrain, lambda^2 per band):** grass 0.01, gravel 0.55, ICE 0.08, post-ice
gravel 1.47 (fall). Key insight: lambda^2 = recoverable goal-CE = err'S^-1 P err (P=controllable
subspace via model DC gain B). On ICE (mu=0.05) control authority B collapses so the error is
UN-recoverable -> lambda^2 stays LOW (0.08). On gravel (mu=0.40) it's genuinely recoverable -> 0.55
(55x grass). So the DT trigger is correctly SELECTIVE (fires where adapting helps), but the fixed
control-cost budget tau=2.28 is too high to catch the real gravel signal (0.55) — only the pre-fall
spike (62) crosses it. NOT blindness; miscalibrated threshold.

**Fix = CusumDecisionMonitor** (subclass of DecisionTheoreticMonitor in experiment-flat2slope-adapt/
run_experiment.py; wired into natural via --trigger cusum). Same lambda^2, Page's CUSUM threshold:
S_k = max(0, S_{k-1} + (lambda^2/tau - kappa)); fire when S>h. Reading lambda^2/tau as per-step rate
of value forgone by holding incumbent, S = cumulative benefit of adapting; fires when it exceeds
adapting cost. Defaults DT_CUSUM_SLACK kappa=0.10, DT_CUSUM_H h=5.0 (tau units). Arm logic: accumulate
only while armed; on fire disarm + reset S=0; re-arm when r<kappa (suboptimality back to baseline) so
it re-fires per transition. _baseline_mean redefined to tau*kappa so shared squash logic pauses when
windowed ratio < kappa. Saves cusum_s trace. CLI: --cusum-slack/--cusum-h; args threaded via job tuple.

**Seed-0 result:** CUSUM fires t=8.0s on the FIRST gravel transition (y=4.6m) — 25s earlier than DT
(t=33.4s at the fall). Re-arms across ICE (lambda^2 low), re-fires t=33.3s on post-ice gravel. n_fires
2 vs DT's 1. noadapt meanJ -2.00 (DT) -> +0.68 (CUSUM) since scoring now brackets the informative
trajectory not just the fall window. Robot still falls at 20.3m (ice-caused, unavoidable), but the
trigger now gives adaptation a real early chance.

**10-seed DT vs CUSUM (hard terrain):** CUSUM fires more (2.7-4.2/run vs DT 1.2-1.5). marxefe 6/10 vs
noadapt 7/10 (looked promising); grid/bo WORSE (10/10, 9/10 — early triggers amplify black-box trials).

**SETTLED at 100 seeds (2026-07-11, CUSUM, hard terrain, feedback ON, manifest_cusum.csv):**
noadapt 71/100 falls (28.8m), grid 98/100 (20.4m), bo 94/100 (20.6m), marxefe 69/100 (30.4m).
- **marxefe vs noadapt: NOT significant** — 69 vs 71 falls (saves 11 loses 9, McNemar p=0.82); dist +1.6m
  median +0.0m (Wilcoxon p=0.11). The n=10 edge was NOISE. Adaptation does NOT beat holding the incumbent.
- **marxefe vs grid/bo: STRONGLY significant** — saves 29/25 loses 0, McNemar p<0.001; dist +10m p<0.001.

CONCLUSION (well-powered, consistent with [[track1-marxefe-no-adaptation]], [[attitude-feedback-subsumes-adaptation]]):
online CPG param adaptation does NOT improve on no-adapt even with credible controller + hard terrain +
early CUSUM trigger. MARX-EFE's real, significant value = ROBUST CAUTION: matches no-adapt while black-box
grid/BO are catastrophically destructive online (94-98% falls); CUSUM's frequent early triggers WIDEN that
gap (more triggers = more destructive physical trials for grid/bo, none for model-based marxefe).
Honest paper framing: contributions are (a) the principled CUSUM suboptimality trigger, (b) model-based
selection is safe online where black-box search is not — NOT "adaptation improves locomotion".
Analysis: experiment-natural-adapt/significance.py --trigger cusum.
