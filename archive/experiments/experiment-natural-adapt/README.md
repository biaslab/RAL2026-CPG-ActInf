# experiment-natural-adapt

Natural-landscape online CPG adaptation, triggered by prediction error.

Applies the **same** triggered / squash-to-stop protocol as
`experiment-flat2slope-adapt` (a single flat→slope change) to a much harder
scenario: a **natural transect with many terrain changes**. The robot walks
forward across a random sequence of forward bands — **grass / gravel / rocks /
river** — each with its own friction *and* its own geometry (grass: gentle
undulation; gravel: fine roughness; rocks: scattered low bumps; river: a
slippery depression). Terrain from `methods.terrain.sample_natural` (the natural
generator recovered from `archive/experiments/experiment-natural`).

## What is reused vs. new

The monitors (`TriggerMonitor`, `DecisionTheoreticMonitor`), the online methods,
the revert-to-best `Safeguard`, and the windowed stability objective
`j_stab_window` are **imported unchanged** from
`experiment-flat2slope-adapt/run_experiment.py` (single source of truth). This
module only swaps:

- **terrain** — `sample_natural` instead of a single ramp, with per-step ground
  friction updated to the band the robot currently stands on
  (`terrain.apply_dynamic_friction`);
- **fall check** — terrain-relative tip-over + an absolute base-height floor
  (valid because the band geometry is shallow, unlike a climbed slope);
- **run loop** — one long **continuous bout** with **no fixed post-trigger
  horizon**: the monitor re-fires at every band transition and the method
  re-adapts (and squashes/pauses between transitions), until a fall, the
  duration cap, or the end of the transect;
- **metrics** — **survival distance** (distance travelled before a fall / the
  terrain end) is the discriminating metric on natural terrain.

## Scenario

1. The robot starts on grass with the **flat/grass-optimal** CPG gait
   (`experiment-flat/results/selected_params.json` — grass μ≈0.70 is essentially
   flat, so the flat optimum is the natural anchor).
2. The MARX-EFE goal-cross-entropy monitor runs every step; normalised to a
   grass-prefix baseline the prediction error is ≈1 on the opening grass and
   rises at each band transition.
3. Each **rising edge above `K`** fires an event; the monitor re-arms once the
   error falls back, so it **fires repeatedly, once per transition**.
4. **Squash-to-stop:** a method adapts only *while* its error is above `K`, and
   pauses once a full window's mean ratio drops back below `K` — so it re-adapts
   at every band and rests in between. The shared safeguard protects the
   destructive optimisers.

**Arms:** `noadapt` (hold the flat/grass gait — anchor), `grid`, `bo`,
`marxefe`. The flat→slope `oracle` anchor is **dropped**: the optimum shifts band
by band, so there is no single oracle gait.

Optional `--trigger dt` uses the decision-theoretic Newton-decrement statistic
(vs. a fixed control-cost budget) instead of the empirical cross-entropy ratio;
its effective threshold is 1.0 (see the parent experiment).

## Outputs

- `results/runs/nat_seed{S}_{method}.npz` — full per-step signals: `t, x, y, z,
  vx, vy, roll, pitch, upright` (robot state), `ce_raw, ce_ema, ratio`
  (prediction error), `applied` (8×N applied CPG parameters), `adapting`
  (proposing vs. paused), plus `window_scores`, `selected_params`, `fire_steps`,
  `trigger_step`, `fall_step`, `reached_end`, the band/zone layout, and config.
- `results/manifest.csv` — one scalar-metric row per run (survival `dist`,
  `fell`, `bands_crossed`, `n_triggers`, `n_pauses`, `mean_J`, tip-over, …).
- `results/config.json` — the run configuration.

## Usage (from repo root)

```bash
python experiment-natural-adapt/run_experiment.py run \
    --seeds 10 --arms noadapt grid bo marxefe --workers 10
python experiment-natural-adapt/analyze.py            # table + summary + traces
python experiment-natural-adapt/analyze.py --seed 3   # trace figures for seed 3
```

`--K` sets the cross-entropy trigger threshold (default 2.0). Each seed samples a
fresh transect (`sample_natural(seed)`), so seeds vary both the band order and
the initial-state jitter.

`analyze.py` prints the per-method survival/stability table and writes
`results/figures/`: `summary.png` (survival distance / falls / bands / speed /
tip-over bars), `velocity_trace_seed{S}.png`, `attitude_recovery_seed{S}.png`,
and `trigger_trace_seed{S}.png` (the MARX-EFE error ratio re-firing per band).

## Notes

- Per the terrain-feasibility envelope, an open-loop CPG has no stable gait on
  steep inclines/declines; the natural transect deliberately uses **friction
  contrast + shallow geometry** (rocks ≤5 cm, river ≈−7 cm), which stays inside
  that envelope, so differences come from adaptation rather than impassable steps.
- On natural terrain almost everything eventually falls, so survival distance —
  not a fixed-horizon fall rate — is the discriminating metric.
```
