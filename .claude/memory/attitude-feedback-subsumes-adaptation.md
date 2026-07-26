---
name: attitude-feedback-subsumes-adaptation
description: "with VMC attitude feedback ON + refit optima, the 10-15° slope stops triggering/falling, so parameter adaptation is largely unnecessary there — undercuts the paper thesis; needs a harder regime"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f8f0747-1c9d-4007-96d6-8da8b667cf75
---

After adding VMC attitude feedback ([[cpg-no-attitude-feedback]]) and re-deriving optima with
feedback ON (10° slope oracle: V -0.38/3-of-8-falls OPEN-LOOP -> V 0.87/0-falls with feedback;
flat 0.97->0.98), ran flat->slope ON-vs-OFF, both triggers, all arms, slopes 10/15°, 20 seeds
(2026-07-10). Ablation = --no-attitude-fb toggles feedback holding the (new) optima fixed.

**Falls% (feedback ON / OFF):**
CE trig, 10°: noadapt 0/75, oracle 0/75, grid 0/95, bo 0/95, marxefe 0/85.
CE trig, 15°: noadapt 5/100, oracle 5/100, grid 60/100, bo 60/100, marxefe 10/100.
DT trig, 10°: all 0/90. DT trig, 15°: noadapt 5, oracle 10, grid 40, bo 40, marxefe 15 (/100).
Trigger fires (ON): 10° 0/100 (feedback absorbs the incline, NO mismatch to detect); 15° 85/100 CE,
45/100 DT. OFF triggers ~100% (unstable immediately).

**THREE load-bearing takeaways:**
1. Attitude feedback is decisive: 10° becomes trivial (0% falls, never triggers), 15° safe arms 5%.
2. **It largely SUBSUMES parameter adaptation at 10-15°:** at 10° adaptation never even fires; at 15°
   no-adapt (5%) already = oracle (5%) and beats marxefe (10%); grid/bo still destructive (40-60%).
   So with a competent controller, online CPG param adaptation gives ~no benefit here. This UNDERCUTS
   the paper's core contribution ([[track1-marxefe-no-adaptation]], [[stability-goal-prior-breakthrough]]
   were about the OPEN-LOOP regime).
3. OFF baseline is unfairly bad: it runs the feedback-TUNED optima open-loop (falls 75-95% even at 10°,
   vs the ORIGINAL open-loop-tuned system's ~40% noadapt at 10°). For a fair "feedback vs original
   system" figure, re-run OFF with the OLD archived optima (archive/results-pre-attitude-2026-07-10).

**NATURAL transect, feedback ON, DT, 10 seeds (2026-07-11) — CONFIRMS subsumption:** all 4 arms
IDENTICAL: 4/10 falls, 6/10 reach the full 44 m end, mean dist ~38 m, tip 1.1° (was ~2.2° open-loop),
trigger fires only 0.5/run. The SAME 4 seeds (1,5,6,9) fall for every arm — adaptation changes nothing.
3 of 4 falls are on the RIVER band (μ=0.20 ice-like); the 6 survivors never trigger (feedback handles
grass/gravel/rock). So on natural terrain too, with the credible controller, online CPG-param adaptation
is INERT; remaining failure is localized to the slippery river, a friction regime no fixed gait+feedback
survives and that 1-2 windows of param tuning can't rescue (DT fires too late anyway).

User decision (2026-07-11): KEEP the credible (feedback-ON) controller; drop the feedback-OFF case and
the simple flat2slope setting; stick with the DT trigger (more principled).

HARD terrain variant (2026-07-11, terrain.py): made natural transect harder on all 3 axes at once —
(1) harsher friction: grass .70->.55, gravel .55->.40, rocks .95->.80, river .20->.10, + new "ice"
band mu=0.05; mix biased 55% to river+ice (weights grass/gravel/rocks/river/ice = .15/.15/.15/.30/.25).
(2) rougher geometry: amplitudes ~2x (rocks 5->11cm, river 7->13cm). (3) per-band inclines: ~half of
non-flat bands (river/ice stay flat) are 8-15° ramps, uphill-biased 2:1 but net-bounded to ~±1.8m via
sign flip when elev>1.5m. New sample_natural returns band_slopes + band_elev0; new terrain.natural_elev_at(cfg,y);
_natural_height_grid adds the incline baseline; river dip carved into baseline. Natural experiment fall
check now terrain-RELATIVE (clearance above natural_elev_at, CLEAR_FALL=0.22; was absolute Z_FALL).

1-seed check (seed 0, DT, feedback ON, all 4 arms IDENTICAL): now FALLS at 20.3m/t=33.6s after 3 bands
(vs reaching 44m end on the easy terrain) — terrain is genuinely harder. But **DT trigger fires exactly
ONCE, at t=33.4s / 0.2s BEFORE the fall** (ratio 1.17), NOT at the ice crossing. During the ice band
(mu=0.05, the destabilizer) the ratio peaked only 0.24, <<1.0; ratio<1.0 for 99% of the trial. Confirms
[[lambda2-trigger-fails-detection]] on hard terrain: the Newton-decrement DT trigger detects the TOPPLE,
not the low-friction transition, so adaptation gets nothing actionable before it's too late. Open: the DT
trigger may be fundamentally unable to give the adaptation story a chance; consider whether the paper needs
the CE trigger, an earlier/faster detector, or a reframe. Not yet resolved.
