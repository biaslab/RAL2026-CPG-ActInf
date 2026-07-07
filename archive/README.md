# Archive

Superseded but potentially-useful material, kept out of the active pipeline.

- `experiments/` — earlier experiment modules (flat, sloped, friction, natural,
  vswitch). Not part of the current paper pipeline; the friction/natural code may
  be a useful reference for the chaotic-terrain experiment. Their old result
  folders were discarded.
- `scripts/`
  - `run_crossover.py` — old flat-vs-sloped crossover analysis (superseded by
    `experiment-flat2sloped/run_oracles.py`).
  - `gen_terrain_optima.py` — per-terrain BO optima generator (produced the
    steeper/decline optima; superseded by `run_oracles.py` for flat/sloped).
  - `make_fig1.py`, `make_snapshots.py` — generators for schematic figures no
    longer used in the paper.
  - `cpg_optima_by_parameter.csv` — optima table consumed only by the above.
- `demo-speedbump/` — speed-bump dashboard demo (video + frames).

Active pipeline lives in `methods/`, `experiment-eventtrigger/`,
`experiment-flat2sloped/` (`run_experiment.py`, `run_oracles.py`,
`make_problem_figures.py`), and the paper figures in `figures/`.
