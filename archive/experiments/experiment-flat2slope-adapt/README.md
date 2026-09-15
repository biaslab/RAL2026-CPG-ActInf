# experiment-flat2slope-adapt

Flat → variable-slope online CPG adaptation, triggered by prediction error.

Combines the two parent experiments into the `problem-statement/` scenario:

- **from `experiment-flat2sloped`** — the flat→slope terrain change and the flat
  / slope CPG optima (`experiment-flat/results/selected_params.json`,
  `experiment-sloped/results/selected_params.json`);
- **from `experiment-eventtrigger`** — the MARX-EFE goal cross-entropy trigger,
  the online adaptation methods (grid / BO / MARX-EFE) with the shared
  revert-to-best safeguard, and the **squash-to-stop** protocol.

## Scenario

1. The robot walks on flat ground with the **flat-optimal** CPG gait. **Five
   metres in** the terrain rises at a slope of variable degree (`--slopes`).
2. A MARX-EFE agent monitors the **prediction error** every control step — the
   goal cross-entropy between its one-step posterior predictive and a
   stable-walking goal prior. Normalised to a flat-walking baseline it is ≈1 on
   the flat prefix and rises on the incline.
3. When that ratio exceeds `K` for ≥2 steps an **event** fires and an online
   method starts searching for new CPG parameters, window by window (1.5 s
   windows, 0.3 s ramps).
4. **Squash-to-stop:** adaptation runs only *while* the error is above `K`. Once
   a full window's mean ratio falls back below `K` the method **pauses** (holds
   its parameters); a re-fire of the re-armed monitor resumes it. A method makes
   at least one genuine proposal before it is allowed to pause.

Arms: `noadapt` (hold flat gait) and `oracle` (jump to slope-optimal gait) are
anchors; `grid`, `bo`, `marxefe` are the online searchers.

## Outputs

Unlike `experiment-eventtrigger` (aggregated metrics only), this writes the
**full per-step signals** of every run:

- `results/runs/slope{D}_seed{S}_{method}.npz` — per-step `t, x, y, z, vx, vy,
  roll, pitch, clear` (robot state), `ce_raw, ce_ema, ratio` (prediction error),
  `applied` (8×N applied CPG parameters), `adapting` (proposing vs. paused), plus
  `window_scores`, `selected_params`, `fire_steps`, `trigger_step`,
  `squash_step`, `fall_step`, and the run config.
- `results/manifest.csv` — one scalar-metric row per run.
- `results/config.json` — the run configuration.

## Usage (from repo root)

```bash
python experiment-flat2slope-adapt/run_experiment.py run \
    --slopes 10 15 --seeds 6 --arms noadapt oracle grid bo marxefe --workers 10
# then open and run experiment-flat2slope-adapt/analyze.ipynb
```

`--K` sets the trigger threshold on the baseline-normalised ratio (default 2.0).
The analysis notebook reads the files above and produces the prediction-error,
attitude, θ(t) parameter-trajectory, per-window, and aggregate comparison
figures.

## Notes

- Fall detection is terrain-relative (clearance above the local ground) since the
  absolute-height test is blind once the robot climbs.
- Per the terrain-feasibility envelope, an open-loop CPG has a stable gait only up
  to ~10° incline; at ≥15° even the slope-optimal `oracle` falls, which
  upper-bounds what any parameter switch can achieve there.
