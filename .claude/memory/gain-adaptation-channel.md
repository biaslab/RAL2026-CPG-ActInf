---
name: gain-adaptation-channel
description: pivot to adapting the 4 attitude-feedback gains (not CPG shape params) — ramped per-band gain oracle is the FIRST oracle to beat no-adapt; instant gain switch harmful; gentle ramp essential
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f8f0747-1c9d-4007-96d6-8da8b667cf75
---

After proving no CPG-shape-param oracle beats no-adapt ([[oracle-cannot-beat-noadapt]]), pivoted to
adapting the 4 VMC attitude-feedback GAINS [kp_roll, kd_roll, kp_pitch, kd_pitch] (the balance-loop
channel, continuous / no gait-phase discontinuity). User asked: can we adapt these AT ALL to improve
locomotion across terrains? (2026-07-12)

**Infra:** JointCPG gains are now instance state + set_gains() (methods/marxefe_optimizer.py); defaults
GAIN_DEFAULT=[0.8,0.05,0.5,0.05], bounds GAIN_LOWER/UPPER (kp<=3, kd<=0.3). natural run_trial takes
gain_policy= (fixed 4-vec, or callable(pos_y)->4-vec for the clairvoyant per-band oracle); CPG shape held
at incumbent. experiment-gain-adapt/ceiling.py: `fit` BO-fits per-surface gains, `compare` no-adapt vs
oracle. Optima in results/surface_gain_optima.json.

**Per-surface gain optima (BO, isolated, feedback ON) — large headroom on hard surfaces:**
ice default V-0.05 -> +0.92 (gains [0.71,0.02,0.15,0]); river -0.06 -> +0.91 ([1.4,0.06,0.48,0.02]);
rocks -1.13 -> -0.25 ([1.22,0.3,0,0.045] — turns OFF pitch correction on bumps!); grass/gravel already ok.

**Ceiling on natural transect (20 seeds):**
- no-adapt (fixed default gains): 15/20 falls, 5 end, 26.7m.
- gain oracle INSTANT switch: 18/20 falls (WORSE) — instant gain jump is its own shock.
- gain oracle RAMPED 1s switch: **13/20 falls (BETTER), 7 end, saves 4 loses 2** — FIRST oracle in the
  whole project to beat no-adapt. McNemar(4,2) p~0.69 (n=20, HINT not proof). dist ~same.

**KEY:** gains are the better channel BUT only with GENTLE (ramped ~1s) adaptation; instant switches
harmful. A gain is continuous so a slow ramp fully removes the switch cost, unlike a gait-shape switch
(retains phase disruption) which is why CPG-param oracle never won. This JUSTIFIES the pivot — headroom
that partly transfers to the transect exists in the gains channel.

**SETTLED at 100 seeds (2026-07-12): the n=20 signal was NOISE — ramped gain oracle DOES NOT beat no-adapt.**
no-adapt 72/100 falls (28 end, 28.8m); gain oracle ramped 76/100 (24 end, 27.8m — slightly WORSE). Saves
13 loses 17, McNemar p=0.59; dist -1.0m Wilcoxon p=0.24. The n=20 (13 vs 15, 6 discordant) was underpowered.

**DEFINITIVE CONCLUSION across BOTH channels + many configs:** NO clairvoyant oracle beats no-adapt —
CPG shape params (mixed/regions, abrupt/gentle) NOR feedback gains (instant/ramped), all at n=100 tie or
lose. The large steady-state per-surface headroom (CPG rocks V -1.13->+0.59; gains ice/river/rocks +0.9)
does NOT transfer to the continuous transect. Root cause: failures are TRANSITION-driven; reactive
adaptation (detect change AT/after the boundary, per [[cusum-suboptimality-trigger]] fires ~0.8s in) is
inherently too late, and the only thing that might win — ANTICIPATORY switching before the boundary — is
NOT realizable (needs terrain-ahead knowledge no online method has). So the realizable ceiling fails.

**Recommendation: STOP the adaptation push; consolidate the honest paper.** Defensible contributions:
(a) CUSUM suboptimality trigger [[cusum-suboptimality-trigger]]; (b) model-based selection is SAFE online
where black-box search is catastrophic (grid/bo 94-98% falls, p<0.001) [[oracle-cannot-beat-noadapt]].
NOT "adaptation improves locomotion" — rigorously refuted via oracle upper bounds on two channels.
The VMC attitude feedback [[cpg-no-attitude-feedback]] is the real robustness win and should be framed as
the contribution, with online param/gain adaptation shown (honestly) to add nothing beyond it here.
