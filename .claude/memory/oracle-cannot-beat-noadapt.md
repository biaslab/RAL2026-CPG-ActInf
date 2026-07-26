---
name: oracle-cannot-beat-noadapt
description: "even a clairvoyant per-band oracle with true per-surface optima ties no-adapt on the natural transect — falls are transition-shock + hard-surface, not parameter mismatch; adaptation is the wrong tool"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f8f0747-1c9d-4007-96d6-8da8b667cf75
---

Definitive answer to "can no-adapt be beaten by parameter adaptation?" — NO, established via an
oracle upper bound (2026-07-12, hard natural terrain, feedback ON, CUSUM trigger, 100 seeds).

**Infrastructure:** experiment-natural-adapt/fit_surface_oracles.py BO-fits a per-surface CPG optimum
(grass/gravel/rocks/river/ice) on each surface's REAL geometry+friction+feedback (score = criterion V
via run_trial single-surface, incumbent injected as probe 0). Writes results/surface_optima.json.
Clairvoyant `oracle` arm (added to run_experiment.py, handled directly in run_trial, no trigger)
switches band-by-band to the current surface's optimum, ramped — the UPPER BOUND on any param switch.
significance.py --trigger cusum reports falls + paired McNemar/Wilcoxon vs no-adapt.

**Two facts that resolve the paradox:**
1. Better per-surface gaits DO exist (BO, not LHS — LHS of random gaits was too weak and wrongly said
   incumbent-is-best). Isolated steady-state V: rocks incumbent -1.13 -> BO +0.59 (+1.72!); river
   -0.06 -> +0.90; ice -0.05 -> +0.93; grass/gravel already optimal (incumbent IS grass optimum).
2. YET the oracle switching to those exact gaits TIES no-adapt on the transect: oracle 67/100 falls vs
   noadapt 71 (McNemar p=0.63), dist +0.7m (Wilcoxon p=0.44), reaches END LESS (14 vs 25/100). marxefe
   69 (p=0.82). grid 98 / bo 94 (catastrophic, p<0.001).

**Why headroom doesn't transfer:** oracle SAVES 21 seeds but LOSES 17 — switching params at a transition
is itself a destabilizing perturbation. Falls only partly at boundaries (oracle 42% within 1.5m of a
band edge vs noadapt 31% — the extra are switching-induced); the rest are mid-band on genuinely hard
surfaces where even the better gait sometimes falls. Steady-state per-surface advantage is washed out by
transition shock + irreducible surface difficulty.

**Terrain redesign attempts (2026-07-12) ALL FAIL to make oracle beat no-adapt:**
Added terrain.sample_region_course(seed): few LONG homogeneous hard regions (rocks/ice/river, 14-18m)
separated by benign grass margins — designed so steady-state cost dominates the one switch per region.
- Abrupt ramp (0.3s): oracle WORSE (5/5 falls vs noadapt 3/5; on 2/5 seeds noadapt reaches END and oracle
  FALLS). Switch shock.
- Ramp-speed sweep (n=5): slowing morph 0.3s->1.0s fixes the switch shock (oracle 5/5->3/5 falls, +6m).
  RAMP=100 (1s) adopted as the real fix (0.3s too abrupt for the large param jumps the oracle makes).
- Region course + 1s ramp, n=12: oracle STILL ties/worse — falls 7/12 vs noadapt 6/12, end 5 vs 6, dist
  +2.3m (noise). marxefe 7/12, 29.8m. The n=5 +7m hint shrank to +2.3m.
Root cause confirmed across ALL designs (mixed short bands, long regions, abrupt & gentle switch): the
steady-state per-surface advantage (real, large: rocks V -1.13->+0.59) does NOT transfer because (a)
incumbent+feedback is robust on survivable instances, (b) hard instances kill ANY gait, (c) switching
adds risk ~= cancels benefit. No terrain within the CPG-shape-param paradigm makes adaptation win.

**CONCLUSION (rigorous):** the oracle is the upper bound on parameter-switching policies (perfect terrain
knowledge + true per-surface optima). Since even it can't beat no-adapt, NO realizable param-adaptation
method can. The failure mode is terrain-transition shock + hard-surface falls, which re-tuning CPG
PARAMETERS does not address (and sometimes worsens). Consistent with [[attitude-feedback-subsumes-adaptation]],
[[track1-marxefe-no-adaptation]]. Implication for paper: cannot claim adaptation improves locomotion; the
defensible contributions are (a) the CUSUM suboptimality trigger [[cusum-suboptimality-trigger]], (b)
model-based selection is SAFE online where black-box search is catastrophic (grid/bo 94-98% falls). A
real gain would need a different actuation channel (e.g. anticipatory/transition-aware control, or the
attitude-feedback GAINS as the adapted variable rather than CPG shape params).
