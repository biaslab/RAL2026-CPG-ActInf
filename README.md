# Active inference for event-triggered CPG adaptation (T-RL 2026)

Manuscript and code for the T-RL submission *"Active inference for
event-triggered adaptation of central pattern generator parameters in
quadrupedal robot locomotion"* (repo `biaslab/TRL2026-CPGActInf`).

A Hopf-oscillator **central pattern generator (CPG)** with an attitude virtual
model controller walks a quadruped (Laikago in PyBullet; Petoi Bittle on
hardware). Partway through a long bout, an **event** changes the robot's own
dynamics — a trunk payload shifts off the sagittal plane, or a hind-leg actuator
degrades — and **persists**. The question is *when* to re-tune the CPG gait
parameters and *what* to re-tune them to, without paying for the answer in falls.

The proposed **active-inference agent** answers both from one goal prior: a
prediction-error CUSUM fires the re-tuning, and expected free energy picks the
new gait against a learned model of the body's response, screened by a memory of
gaits that already fell.

## Parameterization and scoring

The joint-space Righetti-style CPG is parameterized by an 8-D vector

```
θ = [coupling_gain, w_swing, w_stance, F_fast, STOP_gain, hip_amp, knee_amp, b]
```

with shared bounds in `methods/cpg_bounds.py`. The continual experiments search
a **reduced** subspace (`FREE_DIMS_PAYLOAD = [0, 1, 3, 4, 7]` — coupling gain,
`w_swing`, `F_fast`, `STOP_gain`, `b`); the frozen dims stay at the incumbent.
Every arm gets the same subspace, so the comparison is head-to-head.

Two scores appear in the repo, and they are not interchangeable:

- **`score_V`** — the saturating speed reward minus RMS tilt. This is what the
  stored bouts in `results/` were run under.
- **Cross-entropy `J`** — the paper's criterion: the cross-entropy from the
  distribution a gait induces over body outputs `y = [vx, vy, pitch, roll]` to
  the Gaussian goal prior, in nats, **lower is better**.
  `methods/crossentropy_score.py` recomputes it from the stored per-step traces
  over the same evaluation windows, so no simulation has to be re-run.

The headline metric is neither: it is **falls per bout**.

## The bout protocol

One long, non-episodic bout on flat ground (`methods/continual_driver.py`). The
robot walks healthy; after a gap the event engages and persists. A CUSUM on the
forward-speed deficit and excess tilt, relative to the pre-event healthy
baseline, detects it and the responder proposes a gait, ramped in. From there:

- the event is **only** reverted when the robot **falls** — it is never
  auto-healed. On a fall the bout records `V_FALL`, folds it into the responder's
  memory, stands the robot upright where it fell, heals the event, and re-engages
  after a random gap;
- if the responder adapts and does not fall, it keeps walking under the event for
  the rest of the bout.

So no-adapt tips over roughly every time-to-fall seconds and racks up falls,
while a good adapter falls rarely. Only the physics (how the event is applied,
how the robot resets) differs between experiments; the detection and adaptation
methodology lives in `methods/` and is shared exactly.

## Arms compared

All seven run through the same driver, the same reduced subspace, and the same
out-of-process worker, so proposal latency costs real simulation steps.

| Arm | File | Role |
|---|---|---|
| `noadapt` | — | Lower bound: keeps the incumbent gait through the event. |
| `grid` | `methods/event_responders.py` (`GridResponder`) | Latin-hypercube grid search; non-adaptive reference. |
| `bo` | `methods/bo_optimizer.py`, `event_responders.BOResponder` | Bayesian optimization (GP surrogate + UCB). |
| `esc` | `methods/event_responders.py` | Extremum-seeking control: dither-and-climb on the measured score. |
| `safegp` | `methods/gp_safe_agent.py` | Safe GP recovery agent — GP map with a fall-avoidance constraint. |
| `oracle` | `methods/event_responders.py` | Upper bound: jumps straight to the per-phase optimum fitted offline. |
| **`aif`** (proposed) | `methods/aif_recovery.py` | **Unified active-inference agent** (below). |

Supporting modules:
- `methods/cpg_controller.py` — the joint-space CPG controllers (`JointCPG`,
  `PerLegCPG`), pure NumPy so the real robot runs the same controller;
  re-exported from `marxefe_optimizer` for existing call sites.
- `methods/cpg_bounds.py` — shared 8-D parameter bounds. Also holds the Cartesian
  foot-trajectory + IK constants (`ALPHA_HOPF`, `PHI_TROT`, `leg_ik`, …) of the
  Zhang et al. IROS 2024 controller, which was laterally unstable on Laikago
  under position control and is **not** part of the comparison.
- `methods/marxefe_optimizer.py` — the matrix-normal-Wishart AR (MARX) belief and
  its EFE solver, reused by `aif_recovery` via `build_marx_agent`.
- `methods/continual_driver.py`, `methods/continual_driver_aif.py` — the shared
  continual-bout run loop (trigger → responder → apply).
- `methods/event_responders.py`, `methods/responder_worker.py` — the arms and the
  out-of-process worker that makes proposal latency cost simulation steps.
- `methods/crossentropy_score.py` — re-scores already-run bouts under the paper's
  cross-entropy criterion, from the stored traces.
- `methods/terrain.py` — pluggable ground (`flat`, `sloped`, `multislope`,
  `friction`); used by the archived terrain experiments.
- `methods/continual_analysis.py` — shared figure/summary code for the notebooks.

### The `aif` agent

`methods/aif_recovery.py` unifies three components under **one** Gaussian goal
prior over `y = [vx, vy, pitch, roll]`:

- a **fast MARX belief** updated every sim step (100 Hz), with the measured joint
  angles as exogenous input and past outputs as the autoregressive part
  (`ar_order=2`, `forgetting=0.99`);
- a **slow, event-triggered GP map** from CPG parameters to outputs, with a
  persistent memory, that proposes a recovery gait by minimizing EFE — and that
  also *feeds* the AR belief, its prediction entering the regressor as a fourth,
  uncertain input block (`Dg=4`). The belief therefore predicts *with* the gait
  map rather than beside it, and learns online how far to trust it;
- a **trigger**: the cross-entropy from the MARX one-step posterior predictive to
  the goal prior, accumulated in a CUSUM (`cusum_kappa=3.0`, `cusum_h=5.0`).

Trigger and control **decouple on the linear velocities**: the control/EFE goal
keeps `vx` tight (so the agent still drives as fast as it can), while the trigger
goal loosens both `vx` and `vy` so it is an *uprightness-only* signal. It quiets
whenever the robot stays level, however slowly and however much it must crab
sideways to compensate an asymmetric fault. Pitch and roll stay tight and shared.

> The `update_every` / `control_prior_scale` knobs documented in earlier versions
> of this README belong to the standalone `marxefe_optimizer` used by the
> archived per-episode terrain runs. They are not wired into the continual
> experiments.

## Repository layout

```
methods/                     shared library (controllers, optimizers, terrain, bounds,
                             continual-bout driver, responder arms, scoring, analysis)
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
template/                    the T-RL / IEEEtran journal template + how-to
archive/                     superseded experiments, scripts, and the RA-L
                             manuscript (archive/paper-ral2026/)
notes/, literature/          working notes and papers
main.tex, references.bib     the manuscript
IEEEtran.cls, IEEEtran.bst   journal class + bibliography style (IEEEtran.cls is
                             vendored from template/, it is not in texmf)
ieeeconf.cls                 conference class, kept for archive/paper-ral2026/
```

Every experiment folder is **self-contained and consistent**:

```
experiment-simulation/experiment-*/
  run_experiment.py    runs every arm over N seeds, one worker per seed, → results/
  fit_*_oracles.py     fits the oracle arm's target + the cross-penalty screen
  analyze.ipynb        thin notebook over methods/continual_analysis.py → figures
  results/             continual_events.csv, continual_summary.csv, logs/*.npz,
                       figures/, PROVENANCE.md
```

Folders do **not** import from each other; each imports only from `methods/`, and
each writes to its own `results/` regardless of the working directory.

## Building the manuscript

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`IEEEtran.cls` is vendored at the repo root because it is not in the system texmf
tree. The author block in `main.tex` is commented out; uncomment it (and the
`\markboth` header) before submission.

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

## Results

### Payload shift — the headline run

100 seeds × 300 s bouts, 7 arms, run 2026-07-26 (700 bouts, no failures). Falls
per bout, mean ± SEM; **lower is better**.

| Arm | Falls / bout | Mean distance under fault [m] | Mean trial distance [m] |
|---|---|---|---|
| `noadapt` | 5.18 ± 0.18 | 8.31 | 22.94 |
| `grid` | 1.49 ± 0.08 | 2.22 | 8.24 |
| `bo` | 1.59 ± 0.09 | 2.97 | 9.22 |
| `esc` | 1.94 ± 0.05 | 2.59 | 9.62 |
| `safegp` | 0.48 ± 0.08 | 1.14 | 4.74 |
| **`aif`** (proposed) | **0.41 ± 0.08** | 0.71 | 3.90 |
| `oracle` | 0.00 ± 0.00 | −5.38 | −2.94 |

**Read this carefully.** The agent falls 3.6–4.7× less than grid search, BO and
ESC (0.41 vs 1.49–1.94), which is the paper's claim of roughly a quarter of their
falls. But it is **statistically indistinguishable from `safegp`** at these error
bars (0.41 ± 0.08 vs 0.48 ± 0.08) — the advantage over the safe-GP baseline is
not established by this run. The distance columns fall with the fall rate because
an arm that stops falling stops being re-launched under a fresh event; they are
not a performance ranking, and the oracle's negative distances show it.

### Leg damage — no valid results

`experiment-simulation/experiment-damage-adapt/results/` holds only the fitted
inputs (`incumbent.json`, `damage_optima.json`). Everything else predated the
out-of-process responder worker (commit `0af436c`, 2026-07-25), which changed
`methods/continual_driver.py` enough that the old numbers could not be pooled
with current ones, and was deleted on 2026-07-27. **Re-run before quoting any
per-leg damage number.**

### Real robot — bring-up only

`experiment-real/results/` contains characterization logs from 2026-08-19
(shifter tests, stand tests, walk tests) — not an arm comparison. The hardware
payload-shift experiment has not been run.

### What the archived terrain experiments showed

Superseded by the reframing above, from `archive/experiments/`
(`experiment-flat`, `experiment-sloped`, `experiment-friction`), and kept because
they motivate the current design:

1. **All methods find a good gait on easy/static terrain**, with BO the most
   sample-efficient when the problem is stationary.
2. **Faster within-episode recovery was _not_ supported.** A recovery-time edge
   seen at 10 seeds **did not replicate at 20 seeds** — a reminder to demand
   statistical power before claiming adaptation-rate wins.
3. Steep slopes are **fall-dominated** for all methods (80–90% falls). Geometric
   terrain made falls into transition shocks and saturated the steady-state
   objective, so the oracle never beat no-adapt — which is why the experiments
   moved to a *persistent* change in the robot's own dynamics.

## Environment

Python 3.9 (Anaconda base). Dependencies: `pybullet`, `torch`, `botorch`,
`gpytorch`, `casadi`, `scipy`, `numpy`, `pandas`, `matplotlib`.

`methods/__init__.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` (OpenMP: torch's MKL vs
PyBullet) and forces UTF-8 stdout, so the scripts run on a stock Windows console
without manual environment variables.

> Note: the `aif` and `safegp` arms are compute-heavy (GP refits per event, EFE
> solves per proposal). The 100-seed × 300 s payload run above took 20 workers
> with real-time pacing held (first wave 310 s vs 300 s nominal), so the measured
> compute latencies are meaningful. Grid search and BO are far cheaper.
