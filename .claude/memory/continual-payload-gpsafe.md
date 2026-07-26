---
name: continual-payload-gpsafe
description: "experiment-payload-adapt/run_continual.py — continual repeated-payload-shift recovery with GPSafeRecovery (moved to methods/), CUSUM trigger, restore-after-fall; the working adaptation story"
metadata: 
  node_type: memory
  type: project
  originSessionId: aece00c0-2916-4a97-8f01-c7a9ab6d2bbd
---

Created 2026-07-15 (user request): continual analogue of
experiment-damage-adapt/run_continual.py for the payload shift. Moved
`gp_safe_agent.py` from experiment-damage-adapt/ to **methods/** (shared); fixed
the two damage consumers (run_gpsafe.py, run_continual.py) to
`from methods import gp_safe_agent as gp`. gp_safe_agent was NOT git-tracked.

experiment-payload-adapt/run_continual.py: one 120 s bout, 8 kg payload shifts
off-sagittal (default lat 0.215/back 0.20) recurring every 2-8 s of healthy
walking; GPSafeRecovery (EFE objective, free_dims [0,1,3,4,7] =
coupling/w_swing/F_FAST/STOP/b) proposes a gait from persistent GP memory at
each detected shift; on fall → upright-reset + recenter payload; on survive →
record V, recenter. Prediction-error CUSUM detector (vx deficit + tilt vs
warmup baseline). Reuses pl.attach_payload/shift_pivot/_fallen_flat.

Detector tuning was the hard part (4 fix cycles) — key lessons for anyone
touching it:
- shift schedule GATED on detector `armed` (robot confirmed healthy) so every
  shift lands over a live monitor — else undetected limp = stuck state.
- arm health-gate at 0.70*vx_base (0.85 too strict: per-stride EMA ripple breaks
  the streak, never arms).
- LIVENESS watchdogs so it can't stall: ARM_TIMEOUT 6 s force-arm backstop (3 s
  was too eager → armed mid-recovery-transient → false-alarm cascade, 12/17);
  DETECT_TIMEOUT 3 s force-response for a missed/limp shift.
- false alarms (onset=None) do NOT update the GP (would poison the map with
  centered-load V on a different surface); still logged as a metric.
- defaults grace 2.5 s, detect-kappa 0.20, detect-h 1.8, detect-tau 0.4.

Single-seed 120 s after fixes: 14 events, 1 fall (7%), fall rate first-third
25% → last-third 0% (LEARNING signal present), 6 false alarms remaining. Late-
trial detected events show ~0 s latency + negative V (poor late gaits) — detector
degrades / baseline goes stale late; candidate follow-up. --no-detector gives
idealised react-at-onset (isolates GP-safe learning from detector tuning; damage
folder uses this to argue detection isn't the bottleneck).

3-seed 120 s (tuned defaults grace 2.5/kappa 0.20): 39 shift events, **1 fall
(3%)** total, latency 0.40 s, 16 false alarms. This CRUSHES the fixed-arm cliff
result (noadapt/MARX-EFE ~70-90% falls at 0.215) — GPSafeRecovery survives the
repeated shift almost always by remembering fall regions + safe frontier. BUT
"learning curve" is flat (first-third 0% → last-third 8%) because falls are
already so rare there's nothing to learn down; the win is the low absolute rate,
not a within-trial drop. Caveat: 16/39 false alarms (~40%) still fire on
recovery transients (detector tuning, not the agent). Best V per seed +0.47..
+0.82 (forward-walking recoveries, unlike the braking oracle). Fig
results/figures/continual_recovery.png, events results/continual_events.csv.