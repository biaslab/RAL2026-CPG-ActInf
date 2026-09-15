"""Fit CPG oracles on FLAT terrain.

Bayesian optimization over all 8 CPG parameters, each candidate scored by a 30-s
episode with the stability criterion V (methods.oracle_fit). Independent BO seeds
give a distribution of optima. On flat ground the score is taken after a 1.5-s
transient (no terrain crossing).

Usage (from repo root):
    python experiment-flat/run_oracle.py            # run + select (50 seeds)
    python experiment-flat/run_oracle.py run  [--seeds 50 --trials 100]
    python experiment-flat/run_oracle.py select [--top 5 --reps 8]
Outputs: experiment-flat/results/{oracles.csv, selected_params.json}
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from methods import oracle_fit as of

RESULTS = os.path.join(_HERE, "results")
CFG = of.flat_cfg()
SCORE_FROM_Y = None            # flat: score after the transient, no crossing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["run", "select", "all"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.stage in ("run", "all"):
        of.run_oracle("flat", CFG, SCORE_FROM_Y, RESULTS,
                      a.seeds, a.trials, a.workers)
    if a.stage in ("select", "all"):
        of.select_oracle("flat", CFG, SCORE_FROM_Y, RESULTS,
                         a.top, a.reps, a.workers)


if __name__ == "__main__":
    main()
