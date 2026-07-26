---
name: eventtrigger-experiment-findings
description: Event-triggered online adaptation (grid vs BO vs MARXEFE) — MARXEFE matches oracle early protection; trial-based search is destructive online; stability objective J_stab
metadata: 
  node_type: memory
  type: project
  originSessionId: 1bff1887-bb8c-4a94-9205-5b51562d0f1f
---

`experiment-eventtrigger/` (2026-07-03, 100 paired seeds): prediction-error
trigger (0.5 s MARX rollout EMA, K=16σ, 0 FP / 0 miss, ~0.33 s delay) fires on
flat→10° slope; then online CPG re-tuning with one candidate per 1.5 s window.

Key findings:
- MARXEFE 39% falls/20 s ≈ noadapt 40%, decisively better than grid 83% and
  BO 90% (McNemar ≤ 4e-10). Early phase (10 s): oracle 8, MARXEFE 10 vs
  noadapt 19 (p=0.027 / 0.093), grid/BO 37 — MARXEFE matches oracle
  protection with no prior knowledge of the sloped optimum.
- Trial-based optimizers are DESTRUCTIVE online: exploration windows double
  the fall rate vs not adapting, even with trust region + safeguard.
- Protocol was load-bearing: without a ±0.25-range trust region AND a
  revert-to-best safeguard window (uniform across methods), ALL methods incl.
  MARXEFE fell within seconds and were worse than noadapt.
- Over 20 s, oracle (47%) ≈ noadapt (40%): sustained 10° climbing has a steady
  fall hazard (12-14° sustained roll even in survivors) for every parameter
  set — adaptation buys transition survival, not immunity. Prefer the 10 s
  early window as the primary endpoint.
- New stability objective J_stab (per W. Kouw's requirement: stability
  primary, velocity tolerant): min(vx̄/0.5, 1) − (RMS lp-roll + RMS detrended
  lp-pitch)deg/10, lp = 0.5 s moving average (gait rocking is free), fall=−2.
  Healthy flat walking ≈ +0.95; slope climbing ≈ −0.1..−0.3.
- MARXEFE control agent config: forgetting 0.995, control prior mean =
  incumbent params, EFE re-selection per window; monitor agent separate.

Related: [[terrain-optimum-shift-findings]], [[multistep-error-terrain-detection]].
**Why:** this is the core comparison for the RAL2026 continual-adaptation story.
**How to apply:** report the 10 s early window as primary endpoint; keep the
safeguard/trust-region protocol; don't compare methods on raw 20 s fall rates
alone.
