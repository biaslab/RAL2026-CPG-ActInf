---
name: consolidated-continual-experiments
description: "2026-07 payload-adapt & damage-adapt each merged into one run_experiment.py + analyze.py over 5 arms, sharing methods/ modules; fixed broken flat2slope deps"
metadata: 
  node_type: memory
  type: project
  originSessionId: 442f0b08-3945-4276-8849-c8856d320d5e
---

2026-07-16: consolidated the `run_*` trio in **experiment-payload-adapt** and
**experiment-damage-adapt** into a single `run_experiment.py` + `analyze.py` per
folder, over 5 arms: `noadapt, grid, bo, safegp, oracle` (safegp REPLACED marxefe
as the hero method). Protocol = continual recurring events (payload shift / leg
damage), CUSUM detector, random 2-8 s gaps, ~120 s bout, fall -> reset-at-position
+ revert event -> continue.

Shared code lives in `methods/` (single source of truth, mirrors gp_safe_agent):
- `event_responders.py` -- the 5 arms behind a uniform `propose()/update()`; grid/bo
  search the SAME reduced free-dims as safegp (payload [0,1,3,4,7], damage [0,1,4,5,7]).
- `continual_driver.py` -- the detection+adaptation loop (`run_event_bout`, `BoutConfig`,
  `StepState`); physics differ per folder via a `PayloadPhysics`/`DamagePhysics` object.
- `continual_analysis.py` -- figures (`method_comparison`, `learning_curve`,
  `timeseries_seed0`) + summary table; Okabe-Ito palette, safegp=vermillion, noadapt=gray.

Outputs per folder: `results/continual_events.csv` (one row/event, `method` column),
`results/continual_summary.csv`, `results/logs/<arm>_seed<k>.npz`.

**Why non-obvious:** the OLD run_experiment/run_continual/run_gpsafe were ALL BROKEN
after the cleanup commits deleted `experiment-flat2slope-adapt/` (the f2s single-
source-of-truth with online METHODS/Safeguard/monitors) and `experiment-flat/` (the
incumbent). The consolidation reimplements what it needs and hardcodes the incumbent
`[7.607,13.0498,25,52.4044,0.5,0.1,0.5,10]` (also in each `results/incumbent.json`).

**Persist-until-fall (2026-07-16, corrects an earlier auto-heal design):** the event
PERSISTS once engaged and is reverted ONLY on a fall (fall -> heal + reset in place ->
re-engage after a 2-8 s gap). No auto-heal. `EVAL_HOLD` is now just the trailing window
for scoring a SURVIVING gait's V (memory feedback), not a heal timer. Headline metric =
**falls per bout** (count/seed), NOT per-event fall fraction. Driver = continual_driver
`run_event_bout` (logs `cum_falls`); analysis `fig_falls_over_time` (cumulative falls vs
time) replaced the old learning-curve. Incumbent takes ~10 s of full 22 Nm damage to tip
(3.4-17.8 s across seeds); 18 Nm kills the oracle too, so keep 22 Nm.

**Per-leg CPG damage redesign (2026-07-17):** single-leg torque damage with a GLOBAL
symmetric CPG had NO good regime (asymmetry irreducible -> either incumbent robust or
recovery basin a knife-edge; symmetric damage doesn't fell the incumbent). Fix: split the
hip amplitude into ONE PER LEG -> `methods.marxefe_optimizer.PerLegCPG` (11-D control:
5 global + 4 per-leg hipA + kneeA + b; `expand8`/`expand_box` helpers). The incumbent is
the symmetric gait (all 4 hipA equal) -> still fails; the agent searches the **4 per-leg
hipA** (`FREE_DIMS_DAMAGE=[5,6,7,8]`) and recovers by dropping the weak leg's amplitude
(recovery gait: hipA RL~0, RR~0.06, FL~0.28). Damage force **20 Nm** (was 22). Feasibility:
incumbent falls 4/4, per-leg recovery survives + travels ~8.6 m. Only hipA per-leg (4-D);
+kneeA/+b (8/12-D) dilute the search. Oracle refit by the rewritten `fit_damage_oracles.py`.

**Distance metric fix:** analysis + summary now report **distance UNDER the perturbation**
(sum of per-event onset->fall/end progress), NOT total bout distance -- else a failing arm
is credited for fast healthy walking between falls. `continual_analysis._per_seed` sums the
per-event `dist` from the CSV; run_experiment adds `mean_dist_under_fault`.

Results: **damage (per-leg, 4 seeds, 20 Nm): no-adapt 11.5 falls / 14 m, safegp 6.5 / 31 m,
oracle 0.2 / 47 m** -- adaptation now clearly wins on falls AND dist-under-fault. **payload
(10 seeds): no-adapt 13.8 falls, safegp 0.7, oracle 0** -- wins decisively on falls; BUT the
payload recovery gait is near-stationary (survives the CoM shift by ~walking in place / yaws),
so dist-under-fault favours no-adapt there (payload's clean story is FALLS, not distance;
refit the payload recovery for progress if distance must favour adaptation). safegp's
surviving tilt runs high (wobbly survivors) -- minor. Supersedes [[oracle-cannot-beat-noadapt]]
for the damage case (per-leg control makes adaptation win).

**Still broken / TODO:** `fit_payload_oracles.py` / `fit_damage_oracles.py` still
import the deleted `experiment-flat` -- not needed (oracle arm reads existing
`results/*_optima.json`) but re-fitting requires fixing them. "distance travelled"
= net +Y displacement, which conflates speed with yaw drift under asymmetric load
(safegp goes straightest); consider logging x for path length if that matters.
Supersedes [[payload-shift-experiment]] and [[continual-payload-gpsafe]] workflows.
