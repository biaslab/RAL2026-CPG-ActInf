# Provenance of `results/`

**There are currently no valid (post-async) results for this experiment.**
Everything that was here had been generated before the out-of-process responder
worker landed in commit `0af436c` (2026-07-25), which changed
`methods/continual_driver.py`. Those runs were not comparable with anything
current, so `archive-pre-async-20260725/` (2085 files, ~2.1 GB, mostly
`logs/*.npz`) was **deleted on 2026-07-27**. Re-run before using any per-leg
damage numbers:

    python experiment-simulation/experiment-damage-adapt/run_experiment.py \
        --arms noadapt grid bo esc safegp oracle aif \
        --seeds 100 --duration 300 --jobs 20

(`--duration 300`; the CLI default is 120 s and is not what the archived runs
used. `analyze.ipynb` reads the top level of `results/`, which is why the
current run's CSVs and `logs/` belong there rather than in a named subdir.)

Kept because the run scripts still read them as inputs: `incumbent.json`,
`damage_optima.json`.
