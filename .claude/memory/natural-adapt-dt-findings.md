---
name: natural-adapt-dt-findings
description: experiment-natural-adapt (many-band grass/gravel/rocks/river transect) with DT trigger — field ties on survival; DT fires ~1x/run vs CE ~3-5x
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f8f0747-1c9d-4007-96d6-8da8b667cf75
---

`experiment-natural-adapt/` applies the flat2slope-adapt triggered/squash/safeguard
protocol to a NATURAL transect (many friction+geometry band changes from
`terrain.sample_natural`: grass/gravel/rocks/river). Oracle arm dropped (no single
optimum); incumbent = flat/grass-optimal. Reuses monitors/methods/Safeguard from
[[eventtrigger-experiment-findings]] flat2slope runner via importlib; only terrain,
per-step friction (`apply_dynamic_friction`), fall check, continuous multi-transition
loop, and survival-distance metrics are new. Survival distance is the key metric.

Config knobs live in run_experiment.py: DURATION/REACH/BAND_LEN/START_GRASS. The DT
trigger fires only ~1-1.7x/run on natural terrain (vs CE ratio ~3-5x) — far more selective,
declines most band transitions. User dropped CE K-thresholding in favor of DT.

SHORT bands (BAND_LEN 2.5-4.5, REACH 30, DURATION 45s), 10-seed DT: survival ties ~21-22 m,
falls 2-3/10, meanJ positive (~0.6 safe, ~0.3 grid/bo). Field collapses onto no-adapt.

LONG bands (BAND_LEN 6-10, REACH 45, DURATION 80s, START_GRASS 4.0), 10-seed DT (2026-07-10,
current config): each regime persists ~12-20 s of walking. Robot goes ~34 m but crosses
FEWER bands (~4.6). **Falls now separate the methods**: MARX-EFE 4/10 (best) < no-adapt 5/10
< grid/bo 6/10 (worst). meanJ orders the same (marxefe -0.31 ~= no-adapt -0.27 > bo -0.43 >
grid -0.50; all negative bc long bands = sustained time in hard rocks/river). Distance means
overlap (±11 m seed variance) — fall-count & meanJ orderings are the load-bearing signals.

**Why it matters:** lengthening bands turned the short-band tie into a modest but consistent
MARX-EFE edge (survives longest, falls least; grid/bo adapt destructively and fall most) —
the robust-caution reading ([[track1-marxefe-no-adaptation]],
[[stability-goal-prior-breakthrough]]) holds on natural terrain.
