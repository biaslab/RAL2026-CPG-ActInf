---
name: terrain-optimum-shift-findings
description: Flat→sloped switch experiment (100 paired seeds) — optimum shifts with terrain but J saturates; terrain signal lives in falls and speed
metadata: 
  node_type: memory
  type: project
  originSessionId: 1bff1887-bb8c-4a94-9205-5b51562d0f1f
---

`experiment-flat2sloped/` (created 2026-07-03) tested whether the CPG optimum
shifts from flat to a 10° slope, with 100 paired seeds (identical 10 s flat
prefix, σ=0.002 rad initial joint jitter as the only randomness, prefix gap 0).

Key findings:
- Real crossover in the objective J (v*=0.5 tracking − 0.5·CoT): flat-opt beats
  sloped-opt on flat (7.45 vs 7.23, p=1.9e-9), sloped-opt beats flat-opt on
  slope (6.90 vs ~6.5 + falls, p=1.3e-3). So the optimum shifts — but by only
  ~0.2–0.4 out of ~7 between surviving gaits.
- The terrain contrast lives in (a) transition falls: keep flat-opt 19% vs
  switch 9%/8% (McNemar p=0.064/0.043), and (b) achievable speed (fast climber
  0.75 m/s flat vs 0.64 m/s slope, +1.8 m per 10 s over flat-opt on slope).
- The exp(-err/0.05) tracking reward saturates at its 0.85 cap for any
  competent gait at v*=0.5 → likely why grid/BO/MARXEFE terrain comparisons
  looked unclear. Fix: v* ≥ 0.8 m/s (above 10°-slope max) and/or score falls
  explicitly.
- Caveat: the sloped BO-seed-0 optimum used in demo-speedbump falls 5/8 on a
  sustained 10° slope; per-terrain optima in figures/cpg_optima_by_parameter.csv
  vary strongly in robustness. Related: [[bo-flat-falls-diagnosis]],
  [[multistep-error-terrain-detection]].

**Why:** determines how the RAL2026 comparison should score terrain adaptation.
**How to apply:** when designing/comparing adaptation methods, use fall rate and
speed-demanding targets as primary metrics, not the saturated J.
