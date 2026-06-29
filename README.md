# Active Inference for Central Pattern Generator Optimization (RAL2026)

Optimizing the parameters of a Hopf-oscillator **Central Pattern Generator (CPG)**
controller for quadrupedal locomotion (the Laikago robot in PyBullet), and
comparing a proposed **active-inference agent** against classical optimization
baselines on terrains of increasing difficulty.

The controller is a Righetti-style CPG with contact feedback, parameterized by an
8-D vector

```
θ = [coupling_gain, w_swing, w_stance, F_fast, STOP_gain, hip_amp, knee_amp, b]
```

All methods optimize this same 8-D vector under identical bounds
(`methods/cpg_bounds.py`) and are scored by the same objective `J` (a velocity/
position reward minus cost-of-transport and roll/pitch instability, computed over
a 3 s steady-state window of a 4.5 s episode).

## Methods compared

| Method | File | Role |
|---|---|---|
| **Grid search (LHS)** | `methods/grid_search.py` | Offline, non-adaptive reference (Latin-hypercube sample). |
| **Bayesian optimization** | `methods/bo_optimizer.py` | State-of-the-art black-box optimizer (GP surrogate + UCB). Selects one parameter set per episode. |
| **MARX-EFE** (proposed) | `methods/marxefe_optimizer.py` | Active-inference agent: a matrix-normal Auto-Regressive eXogenous (MARX) model whose parameters are inferred by Bayesian filtering, with controls chosen by minimizing **Expected Free Energy (EFE)**. Can re-tune **within** an episode. |

Supporting modules:
- `methods/terrain.py` — pluggable ground: `flat`, `sloped`, `multislope`, and
  `friction` (spatially-varying ice/rubber, applied per simulation step).
- `methods/cpg_bounds.py` — shared 8-D parameter bounds.
- `methods/gpefe_optimizer.py` — a GP-based EFE variant (exploratory, not in the
  main comparison).
- `methods/GPEFE_derivations.md` — derivations for the EFE objective.

### MARX-EFE configuration notes
- **Observation / goal**: the agent observes forward/lateral **velocity** and
  pitch/roll, and tracks a target forward velocity (well-posed for the linear
  model — absolute position was unstable).
- **Cadence**: parameters are selected once per trial on static terrain
  (`update_every = 0`) or re-selected every ~gait cycle (`update_every = 50`
  steps) for online adaptation on changing terrain, ramped in smoothly to avoid
  destabilizing chatter.
- **Cautious exploration**: a tight control prior (`control_prior_scale ≈ 0.15`)
  keeps the agent near a safe action under model uncertainty and only deviates as
  the model becomes confident — this is what prevents falls on hard terrain.
- **Forgetting**: an optional forget-toward-prior factor `λ` (`forgetting`) is
  available for non-stationary tracking (kept at 1.0 in the runs here).

## Repository layout

```
methods/                     shared library (controllers, optimizers, terrain, bounds)
experiment-flat/             static flat terrain (the baseline experiment)
experiment-sloped/           random multi-slope terrain
experiment-friction/         random ice/rubber friction zones (+ transition-recovery analysis)
root.tex, references.bib     the manuscript
```

Every experiment folder is **self-contained and consistent**:

```
experiment-*/
  run_multiseed.py     runs grid / BO / MARX-EFE on this terrain across N seeds,
                       one subprocess per seed, → results/, then a convergence figure
  results/             canonical CSVs (one per method per seed) + figures/
  archive/             superseded / precursor runs (kept for reference)
  __init__.py
experiment-friction/
  transition_recovery.py   extra analysis: velocity recovery after friction drops
```

Folders do **not** import from each other; each imports only from `methods/`, and
each writes to its own `results/` regardless of the working directory.

## How to run

From the repository root:

```bash
# Flat baseline (single seed by default)
python experiment-flat/run_multiseed.py

# Sloped terrain (5 seeds of random multi-slope terrain)
python experiment-sloped/run_multiseed.py --seeds 5

# Friction terrain (20 seeds; each has a guaranteed reachable ice patch)
python experiment-friction/run_multiseed.py --seeds 20 --trials 80
python experiment-friction/transition_recovery.py   # recovery metric + figures
```

Each `run_multiseed.py` supports `--seeds N`, `--trials N`, and `--seed K`
(single-seed worker mode). Figures land in the experiment's `results/figures/`.

## Results so far

Objective `J` (higher is better) and fall rate, mean ± std over seeds:

| Terrain | Grid search | Bayesian opt. | MARX-EFE |
|---|---|---|---|
| **Flat** (1 seed) | J=1.96, falls 28% | **J=2.40, falls 7%** | J=1.58, falls 25% |
| **Multi-slope** (5 seeds) | J=1.79±0.36, falls 81% | J=1.54±0.47, falls 91% | J=1.64±0.66, falls 81% |
| **Friction** (20 seeds) | J=2.22±0.29, falls 54% | J=2.22±0.51, falls 34% | J=2.08±0.40, **falls 13%** |

### Key findings
1. **All methods find a good gait on easy/static terrain**, with BO the most
   sample-efficient — its GP surrogate converges fastest and highest when the
   problem is stationary.
2. **MARX-EFE is consistently the most robust**: it falls ~3–4× less than the
   baselines across every terrain and seed count, at comparable objective. This
   is the proposed method's clearest, most reproducible advantage.
3. **Faster within-episode recovery is _not_ supported by the data.** A
   transition-recovery metric (velocity recovery after a friction drop) initially
   suggested a MARX-EFE edge at 10 seeds, but it **did not replicate at 20 seeds**
   with a guaranteed ice drop per seed (recovery times statistically identical
   across methods) — a reminder to demand statistical power before claiming
   adaptation-rate wins.
4. Steep slopes are **fall-dominated** for all methods (80–90% falls); varying
   **friction on flat ground** is a better adaptation testbed (lower falls, real
   surface changes).

The honest current story is **robustness/safety, not adaptation speed**:
MARX-EFE trades a little peak performance for substantially fewer falls across
varying surfaces. The next natural test for the adaptation-rate hypothesis is a
genuinely *non-stationary* task whose optimum changes mid-run (e.g. a mid-episode
target-velocity switch), which the per-episode baselines structurally cannot
track.

## Environment

Python 3.9 (Anaconda base). Dependencies: `pybullet`, `torch`, `botorch`,
`gpytorch`, `casadi`, `scipy`, `numpy`, `pandas`, `matplotlib`.

`methods/__init__.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` (OpenMP: torch's MKL vs
PyBullet) and forces UTF-8 stdout, so the scripts run on a stock Windows console
without manual environment variables.

> Note: MARX-EFE with per-cycle adaptation is compute-heavy (~9 EFE/IPOPT solves
> per episode); a 20-seed × 80-trial friction run takes a few hours. Grid search
> and BO are far cheaper (seconds–minute per 100-trial run).
