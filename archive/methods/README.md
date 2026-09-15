# archive/methods

Modules retired from `methods/` on 2026-07-27, when the repo was reorganised
around `experiment-simulation/`. Each was live only for the per-terrain
experiments that now sit in `archive/experiments/`; nothing under
`experiment-simulation/`, `problem/` or `methods/` imports them any more.

| module | was used by | why retired |
|---|---|---|
| `oracle_fit.py` | `experiment-{flat,sloped}/run_oracle.py`, `experiment-flat2sloped/make_oracle_figure.py` | the per-terrain BO oracle-fit engine; the payload/damage experiments fit their oracles in their own `fit_*_oracles.py` |
| `episode.py` | `oracle_fit.py`, `experiment-flat2sloped/`, `scripts/make_paper_figures.py` | the fixed-length single-episode runner; superseded by the continual-bout driver (`methods/continual_driver.py`) |
| `grid_search.py` | `experiment-friction/run_multiseed.py` | the offline LHS grid baseline; the live `grid` arm is `GridResponder` in `methods/event_responders.py` |

**Import caveat.** The archived experiments still say
`from methods.oracle_fit import ...` / `from methods.episode import ...`. Those
imports no longer resolve, because these files are here rather than in
`methods/`. To run one of the archived experiments, copy the module it needs
back into `methods/` for the duration (or add this directory to `sys.path`).

`archive/` is gitignored, so nothing here is version-controlled; the last
tracked revision of each file is in git history under its old `methods/` path.
