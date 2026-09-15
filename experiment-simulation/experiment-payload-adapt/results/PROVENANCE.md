# Provenance of `results/`

**Top-level `continual_events.csv` / `continual_summary.csv` / `logs/`** (what
`analyze.ipynb` reads) come from:

    python experiment-simulation/experiment-payload-adapt/run_experiment.py \
        --arms noadapt grid bo esc safegp oracle aif \
        --seeds 100 --duration 300 --jobs 20

run 2026-07-26, 700 bouts, no failures. This is the first payload run with the
ESC arm and with the trigger that re-arms *within* an event. Real-time pacing
held at 20 workers (first wave 310 s vs 300 s nominal), so the measured
compute latencies are meaningful.

Note `--duration 300`; the CLI default is 120 s and is NOT what these runs use.

**Deleted on 2026-07-27** (superseded, and not comparable with the above):

* `rerun-10seed-esc-readapt/` — the 10-seed pilot of the same configuration,
  superseded by the 100-seed run above (~81 MB).
* `archive-pre-async-20260725/` — everything generated before the
  out-of-process responder worker landed in commit `0af436c` (2026-07-25), which
  changed `methods/continual_driver.py`. Those numbers were NOT comparable with
  the current ones: `noadapt`, whose own code that commit did not touch, sat at
  14.26 falls/bout in that 100-seed run versus 5.18 now — they could never be
  pooled, so the 1517 files (~1.5 GB, mostly `logs/*.npz`) were removed.

Kept because the run scripts still read them as inputs:
`incumbent.json`, `payload_optima.json`. `fit_payload_oracles.py` now takes the
incumbent from `incumbent.json` via `run_experiment.load_incumbent()`; the
original flat BO fit it was copied from is at
`archive/experiments/experiment-flat/results/selected_params.json`.
