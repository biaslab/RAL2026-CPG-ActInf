---
name: cpg-terrain-feasibility-envelope
description: The open-loop joint-space CPG is viable only on flat + ~10° incline; ≥15° incline and 10° decline have no stable gait (all BO optima fall 8/8)
metadata: 
  node_type: memory
  type: project
  originSessionId: 5f16b388-ca3e-4120-94fd-036b235daf4c
---

Track-2 (RA-L M6 generalization) offline-BO optima generation
(`experiment-flat2sloped/gen_terrain_optima.py`, 5 BO seeds × 100 trials per
terrain, run 2026-07-05) found NO stable gait for steeper inclines or declines:

- incline15, incline20: every BO seed's optimum falls 8/8 in preselection,
  ~0.0 m traveled — the robot cannot climb ≥15°.
- decline10: every BO seed's optimum falls 8/8, ~2.9 m traveled — descends a
  bit then tumbles (CPG tuned for flat/uphill can't descend stably).
- Only flat and the original ~10° incline are feasible (and even θ*(10°) falls
  47% over 20 s in the event-trigger experiment).
- friction drop (flat→ice μ=0.15): a NON-EVENT. Diagnostic (2026-07-05): on ice
  vx 0.522→0.521, |roll| 0.20→0.17°, no fall, trigger z-score max −0.33 (never
  fires; calibrate = 8/8 misses at every K). The legs are POSITION-controlled,
  so commanded joint angles are followed regardless of ground grip → the robot
  is insensitive to friction. Friction adaptation is moot on this platform.

**Why:** the open-loop joint-space CPG (no closed-loop attitude control) has a
narrow terrain envelope. This is the reviewer's M6 "intrinsic fragility of the
open-loop CPG" point, made concrete — controller fragility, not the adaptation
method, dominates outside a ±~10° band.

**How to apply:** do NOT frame Track-2 generalization around steeper/decline
inclines — there is no valid oracle there. Use qualitatively-different but
FLAT-risk terrains instead: friction drop (flat→ice; friction optimum already
in figures/cpg_optima_by_parameter.csv, needs apply_dynamic_friction added to
the event-trigger run_trial loop) and repeated ≤10° transitions
(flat→10°→flat; needs TriggerMonitor re-arming). The infeasibility itself is a
reportable M6 result. See [[track1-marxefe-no-adaptation]]. The non-viable
incline15/20/decline10 rows in cpg_optima_by_parameter.csv and their
selected_params_*.json are kept only as provenance (they all fall).
