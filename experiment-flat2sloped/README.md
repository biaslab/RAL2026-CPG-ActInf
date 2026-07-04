# Flat→sloped terrain switch: does the CPG optimum shift with terrain?

Paired 100-seed experiment testing whether the BO-optimal CPG parameters for
flat terrain remain optimal when the terrain becomes a 10° incline.

## Design

- The Laikago walks with the **flat-optimal** parameters on flat ground for
  10 s. A per-seed calibration run (slope pushed out of reach, otherwise
  identical) measures the base position y₁₀ at t = 10 s; the main runs place
  the ramp start exactly at y₁₀, so the terrain "shifts to sloped" at t = 10 s.
- Paired conditions per seed, **bit-identical for the first 10 s**
  (verified: max prefix gap = 0):
  - `keep` — hold the flat-optimal parameters,
  - `switch` — ramp to the sloped-optimal parameters over 0.4 s at t = 10 s,
  - `switch_fast` — ramp to the fastest-climbing sloped candidate.
- **No randomization over starting points**: every run starts from the same
  nominal pose at the origin. Seeds differ only through a σ = 0.002 rad initial
  joint-angle jitter, which the chaotic contact dynamics amplify.
- Parameter sets were pre-selected from `figures/cpg_optima_by_parameter.csv`
  (5 BO seeds per terrain) by fewest falls, then highest mean objective J on
  their own terrain (8 reps): flat = BO seed 2, sloped = BO seed 2,
  fastest climber = sloped BO seed 1. Note: the sloped BO-seed-0 optimum used
  in demo-speedbump falls 5/8 times on a sustained 10° slope.
- Metrics over the post-switch window t ∈ [10, 20] s; J = velocity-tracking
  reward (v* = 0.5 m/s) − 0.5·CoT, as in `methods/bo_optimizer.py`;
  falls scored J = −50.

## Results (100/100 valid seeds)

| post-switch, on slope      | keep (flat-opt) | switch (sloped-opt) | switch_fast |
|----------------------------|-----------------|---------------------|-------------|
| fall rate                  | **19 %**        | 9 %                 | 8 %         |
| forward distance (median)  | 4.49 m          | 4.78 m              | **6.24 m**  |
| J among no-fall pairs      | 6.87            | 6.82                | 4.49        |

- Falls: McNemar p = 0.064 (switch), p = 0.043 (switch_fast).
- Distance: switch better in 91/100 seeds (Wilcoxon p = 1.5e-10),
  switch_fast better in 93/100 (p = 3.5e-15).
- J among surviving pairs: keep ≈ switch (p = 0.07) — the tracking reward
  saturates at its 0.85 cap for any competent gait at v* = 0.5 m/s.

**Crossover check** (`run_crossover.py`, 3 sets × 2 terrains × 32 reps,
10 s episodes from standstill, slope at 2 m):

| J (mean ± std)  | flat            | sloped          |
|-----------------|-----------------|-----------------|
| flat-opt        | **7.45 ± 0.12** | 4.71 ± 9.84¹    |
| sloped-opt      | 7.23 ± 0.05     | **6.90 ± 0.18** |
| sloped-fast     | 2.48 ± 0.23     | 4.03 ± 0.17     |

¹ one fall (J = −50) in 32; median ≈ 6.5.
Flat-opt > sloped-opt on flat (p = 1.9e-9); sloped-opt > flat-opt on sloped
(p = 1.3e-3): a statistically significant crossover in J in both directions.

## Conclusions

1. **The optimum does shift with terrain** — the crossover in J is real and
   significant in both directions, and terrain-matched parameters roughly
   halve the fall rate at the flat→slope transition.
2. **But the shift is nearly invisible in J** (~0.2–0.4 out of ~7 between
   surviving gaits): the exponential tracking reward saturates because
   v* = 0.5 m/s is comfortably achievable on both terrains. The terrain
   contrast lives almost entirely in (a) transition falls and (b) achievable
   speed (fast climber: 0.75 m/s flat, 0.64 m/s slope), neither of which the
   saturated objective rewards. This plausibly explains why the
   grid/BO/MARX-EFE comparisons across terrains looked unclear.
3. Suggested fixes for the main comparison: raise v* to a demanding value
   (≥ 0.8 m/s, above what a 10° slope allows) so terrain forces a genuine
   trade-off, and/or score falls explicitly rather than through truncation.

## Usage

```bash
python experiment-flat2sloped/run_experiment.py preselect   # pick param sets
python experiment-flat2sloped/run_experiment.py run         # 100-seed main run
python experiment-flat2sloped/run_experiment.py aggregate   # stats + figure
python experiment-flat2sloped/run_crossover.py              # 3x2 crossover matrix
```

Outputs in `results/`: `selected_params.json`, `switch_experiment.csv`,
`switch_comparison.png`.
