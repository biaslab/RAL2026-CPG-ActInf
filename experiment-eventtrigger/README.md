# Event-triggered CPG adaptation: GridSearch vs BO vs MARX-EFE, online

Follow-up to `experiment-flat2sloped/`, which established that (a) the CPG
optimum shifts with terrain, and (b) the cost of a terrain mismatch is paid in
*transition falls*, not in the saturating velocity-tracking objective.
This experiment therefore tests **event-triggered online adaptation**: when a
prediction-error monitor detects the terrain change, an optimizer starts
re-tuning the CPG parameters *in the same run*, and the hypothesis is that
the method that finds better parameters faster remains more stable — the
longer the robot walks the slope with mismatched parameters (sustained roll),
the harder the correction.

## Stability objective (replaces the saturating velocity-tracking J)

Scored per 1.5 s window:

    J_stab = min( mean(vx) / v* , 1 )  −  ( RMS(lp roll) + RMS(lp pitch − median) ) [deg] / 10

with v* = 0.5 m/s and a fall scoring −2. Design rationale (requirements from
W. Kouw): stability is primary; velocity may deviate (slower uphill, faster
downhill) as long as locomotion continues toward the target. Hence the
progress term saturates at v* (faster is not rewarded, slower is only mildly
penalized) and the attitude term uses a 0.5 s moving average ("lp") so the
CPG's natural stride-frequency rocking is free — only *sustained* roll lean
and slow pitch drift are penalized. Pitch is median-detrended per window so
the slope's static pitch is not.

## Trigger

The 0.5-s-ahead MARX rollout-error monitor from `demo-speedbump` (one-step
innovations do not register terrain changes; the 50-step rollout error spikes
reliably). EMA (α = 0.08) compared against a per-run flat baseline
(t ∈ [2.4, 3.4] s); fires at `baseline_mean + K·baseline_std`.

Calibration (`calibrate` stage, 20 seeds): K = 16 is the smallest threshold
with **0 false positives** in 30 s flat-only runs and **0 missed detections**
on flat→slope runs; mean detection delay +0.33 s after the base crosses the
slope start (max 1.31 s; occasionally fires slightly early because the front
feet reach the ramp ~0.3 m before the base does).

## Arms (all share a bit-identical pre-trigger prefix per seed)

| arm      | behaviour after the trigger                                        |
|----------|--------------------------------------------------------------------|
| noadapt  | keep flat-optimal parameters (lower anchor)                        |
| oracle   | ramp straight to the sloped-optimal parameters (upper anchor)      |
| grid     | Latin-hypercube candidate sequence (repo grid baseline), 1/window  |
| bo       | GP-UCB (`methods.bo_optimizer.BOOptimizer`) on windowed J_stab     |
| marxefe  | EFE minimization under the online MARX posterior (updated every    |
|          | control step from t = 0, forgetting 0.995, control prior centred   |
|          | on the incumbent)                                                  |

Candidates are applied one per 1.5 s window (0.3 s ramp). All three
optimizers run inside the same **safeguard layer**: the best-known
(parameters, window-J) pair — initialized with the incumbent and its last
pre-trigger window score — is restored for one window whenever a candidate
scores more than 0.15 below it. Recovery windows are scored and fed back to
the optimizer like any other window. Without this layer (pilot, in git
history of this file's results): *every* method falls within seconds — raw
1-candidate-per-second exploration is lethal mid-walk regardless of method,
and all methods are then worse than not adapting at all.

Search space: a trust region of ±0.25 × (bounds range) around the incumbent,
clipped to the shared bounds — identical for grid, bo and marxefe
(`--trust-radius 0`, full bounds, reproduces the everyone-falls-immediately
regime). Pre-selection of the incumbent (flat-optimal) and oracle
(sloped-optimal) vectors comes from `experiment-flat2sloped`.

## Protocol details

- Per seed: slope start placed at the robot's t = 10 s position (per-seed
  calibration run); σ = 0.002 rad initial joint jitter is the only source of
  across-seed variability; no randomization over starting points.
- Post-trigger horizon 20 s; episode ends at a fall (terrain-relative height
  criterion + orientation criterion).
- The trigger monitor is identical in every arm, and all arms are identical
  until the trigger, so the event fires at the same step across arms
  (`trigger_spread` column verifies this).

## Results (100/100 valid seeds, trigger spread 0 steps across arms)

Trigger fired at 10.34 ± 0.27 s (slope reached at 10.0 s).

| post-trigger              | noadapt | oracle | grid | bo  | marxefe |
|---------------------------|---------|--------|------|-----|---------|
| falls within 20 s         | 40 %    | 47 %   | 83 % | 90 %| **39 %**|
| falls within 10 s (early) | 19 %    | **8 %**| 37 % | 37 %| **10 %**|
| mean survival [s]         | 16.7    | 17.3   | 12.3 | 12.2| **17.8**|
| mean J_stab               | −0.36   | −0.30  | −0.54|−0.54| −0.33   |
| forward distance [m]      | 7.01    | 7.80   | 5.40 | 5.27| 7.41    |

- **MARX-EFE vs grid/BO: decisive.** Falls 39 % vs 83 %/90 %
  (McNemar p = 3.9e-10 / 8.6e-14; survival Wilcoxon p ≈ 1e-12). The
  trial-based optimizers must spend 1.5 s windows walking on speculative
  candidates, and even with the trust region and the revert-to-best safeguard
  that exploration *doubles* the fall rate relative to doing nothing
  (p < 1e-11). MARX-EFE selects from its continuously-updated model without
  behavioural experiments and is the only method that adapts without paying a
  stability price.
- **Early phase confirms the hypothesis' mechanism.** Within the first 10 s
  after the trigger — the window where the terrain mismatch itself is the
  hazard — falls are: noadapt 19, oracle 8 (p = 0.027 vs noadapt), marxefe 10
  (p = 0.093, and p < 1e-5 vs grid/bo at 37 each). MARX-EFE matches the
  oracle's protection *without prior knowledge of the sloped optimum*: it
  finds a good correction within ~1-2 windows.
- **Over the full 20 s the anchors converge** (noadapt 40 %, oracle 47 %,
  n.s.): every fixed parameter set has a steady fall hazard while climbing
  (sustained roll excursions of 12-14° even for survivors), so the early
  advantage of a correct switch is diluted by later, terrain-unrelated falls.
  Sustained 10° climbing is intrinsically precarious for this open-loop CPG;
  parameter adaptation buys transition survival, not immunity.
- Without the safeguard + trust region (pilot configurations), *all* methods
  including MARX-EFE fell within seconds and were far worse than noadapt —
  the protocol design (candidate exposure limiting) matters as much as the
  optimizer.

## Usage

```bash
python experiment-eventtrigger/run_experiment.py calibrate            # trigger K
python experiment-eventtrigger/run_experiment.py run --seeds 100 --trust-radius 0.25
python experiment-eventtrigger/run_experiment.py aggregate
```
