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
| **Grid search (LHS)** | `methods/event_responders.py` (`GridResponder`) | Offline, non-adaptive reference (Latin-hypercube sample). |
| **Bayesian optimization** | `methods/bo_optimizer.py` | State-of-the-art black-box optimizer (GP surrogate + UCB). Selects one parameter set per episode. |
| **MARX-EFE** (proposed) | `methods/marxefe_optimizer.py` | Active-inference agent: a matrix-normal Auto-Regressive eXogenous (MARX) model whose parameters are inferred by Bayesian filtering, with controls chosen by minimizing **Expected Free Energy (EFE)**. Can re-tune **within** an episode. |

Supporting modules:
- `methods/cpg_controller.py` — the joint-space CPG controllers (`JointCPG`,
  `PerLegCPG`), pure NumPy so the real robot can run the same controller;
  re-exported from `marxefe_optimizer` for existing call sites.
- `methods/terrain.py` — pluggable ground: `flat`, `sloped`, `multislope`, and
  `friction` (spatially-varying ice/rubber, applied per simulation step).
- `methods/cpg_bounds.py` — shared 8-D parameter bounds.
- `methods/gp_safe_agent.py` — the safe GP recovery agent (`safegp` arm).
- `methods/continual_driver.py`, `methods/continual_driver_aif.py` — the shared
  continual-bout run loop (trigger → responder → apply) used by every experiment.
- `methods/event_responders.py`, `methods/responder_worker.py` — the arms
  (`noadapt`/`grid`/`bo`/`esc`/`safegp`/`oracle`/`aif`) and the out-of-process
  worker that makes proposal latency cost simulation steps.
- `methods/continual_analysis.py` — shared figure/summary code for the notebooks.

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
methods/                     shared library (controllers, optimizers, terrain, bounds,
                             continual-bout driver, responder arms, analysis)
experiment-simulation/       the PyBullet experiments
  experiment-payload-adapt/    8 kg trunk payload shifting off the sagittal plane
  experiment-damage-adapt/     partial actuator failure in one hind leg
experiment-real/             Bittle hardware: the payload-shift experiment
                             (`run_experiment.py` + `bittle_interface.py`) on the
                             SAME driver/arms as the simulation, the vendored
                             Petoi `PetoiRobot/` API, and `petoi_Hopf.py` (the
                             original hand-tuned CPG demo)
problem/                     problem-statement notebook + the Laikago schematic
printing/                    3-D-printable Bittle CoM-shift harness (.scad/.stl)
figures/                     the manuscript's figures (\graphicspath)
archive/                     superseded experiments and scripts (kept for reference)
notes/, literature/          working notes and papers
root.tex, references.bib     the manuscript
```

Every experiment folder is **self-contained and consistent**:

```
experiment-simulation/experiment-*/
  run_experiment.py    runs every arm over N seeds, one worker per seed, → results/
  fit_*_oracles.py     fits the oracle arm's target + the cross-penalty screen
  analyze.ipynb        thin notebook over methods/continual_analysis.py → figures
  results/             continual_events.csv, continual_summary.csv, logs/*.npz,
                       figures/, PROVENANCE.md, and archived pre-async runs
```

Folders do **not** import from each other; each imports only from `methods/`, and
each writes to its own `results/` regardless of the working directory.

## How to run

From the repository root:

```bash
# Payload-shift experiment: fit the oracle target + cross-penalty screen, then run
python experiment-simulation/experiment-payload-adapt/fit_payload_oracles.py --trials 60 --seeds 3
python experiment-simulation/experiment-payload-adapt/run_experiment.py \
    --arms noadapt grid bo esc safegp oracle aif --seeds 100 --duration 300 --jobs 20

# Leg-damage experiment (same protocol)
python experiment-simulation/experiment-damage-adapt/fit_damage_oracles.py --trials 60 --seeds 3
python experiment-simulation/experiment-damage-adapt/run_experiment.py \
    --arms noadapt grid bo esc safegp oracle aif --seeds 100 --duration 300 --jobs 20
```

Note `--duration 300`: the CLI default is 120 s and is not what the reported runs
use (see each experiment's `results/PROVENANCE.md`). Figures come from the
per-experiment `analyze.ipynb` and land in `results/figures/`.

The Bittle experiment runs the same `methods/` driver and arms as the simulation,
with the physics swapped for a serial link (see `experiment-real/README.md` for
the bring-up checklist — IMU signs, harness end stops and control rate must be
measured per robot):

```bash
cd experiment-real
python run_experiment.py --mode rate     # achievable control rate -> --dt
python run_experiment.py --mode imu      # IMU units + roll/pitch signs
python run_experiment.py --mode walk --duration 20      # does the gait transfer?
python run_experiment.py --arms noadapt aif safegp bo --seeds 3 --duration 120
python run_experiment.py --dry-run --no-prompt --arms noadapt safegp  # no robot
```

The vendored `*Example.py` scripts and `petoi_Hopf.py` still need to be run from
inside `experiment-real/` (they do `from PetoiRobot import *`).

## Results so far

> These numbers predate the payload/damage reframing and come from the terrain
> experiments now under `archive/experiments/` (`experiment-flat`,
> `experiment-sloped`, `experiment-friction`). The current headline results are
> falls-per-bout from `experiment-simulation/*/results/`.

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
