---
name: repo-layout-2026-07-27
description: 2026-07-27 repo reorg — experiments regrouped under experiment-simulation/, experiment-real/ added, problem-statement/ renamed to problem/, experiment-flat archived
metadata:
  type: project
---

2026-07-27 the repo was reorganised. Current top-level layout:

- `experiment-simulation/experiment-{payload,damage}-adapt/` — the two PyBullet
  experiments (previously top-level). They are now **two** levels below the repo
  root, so their scripts compute `_REPO = os.path.dirname(os.path.dirname(_HERE))`.
  Their `analyze.ipynb` sets `EXP = os.path.join('experiment-simulation', ...)`.
- `experiment-real/` — Bittle hardware. Vendored Petoi `PetoiRobot/` package plus
  `petoi_Hopf.py` and the `*Example.py` scripts, which do `from PetoiRobot import *`
  and therefore must be **run from inside `experiment-real/`**. Do not rewrite the
  vendored package's relative imports.
- `problem/` — renamed from `problem-statement/` (`problem.ipynb`,
  `laikago_schematic.tex`, referenced from `root.tex` line ~128).
- `printing/` — 3-D-printable Bittle CoM-shift harness, see [[paper-payload-reframe]].
- `archive/experiments/` — every retired experiment, including `experiment-flat`
  and `experiment-sloped`. `problem/problem.ipynb` reads their
  `results/selected_params.json` from there.

`fit_payload_oracles.py` no longer reads `experiment-flat/results/` at all; it
takes the incumbent from its own `results/incumbent.json` via
`run_experiment.load_incumbent()`, which keeps the folder self-contained.

**Why:** several memories and READMEs still narrate the pre-move layout; a file
named at a top-level `experiment-*/` path in an older memory is almost certainly
under `experiment-simulation/` or `archive/experiments/` now.

**How to apply:** resolve any `experiment-*` path from an older memory against
this layout before assuming it is missing. See
[[consolidated-continual-experiments]] and [[esc-baseline-and-readapt]].
