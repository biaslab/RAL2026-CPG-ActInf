"""Fit CPG oracles on the 10-deg SLOPED terrain.

Bayesian optimization over all 8 CPG parameters, each candidate scored by a 30-s
episode with the stability criterion V (methods.oracle_fit). The ramp starts 2 m
in, so the robot climbs for most of the bout; the score is taken from the slope
crossing (forward pitch is detrended = terrain-relative, so a stable climb is not
penalized for following the incline). Independent BO seeds give a distribution of
optima.

Usage (from repo root):
    python experiment-sloped/run_oracle.py            # run + select (50 seeds)
    python experiment-sloped/run_oracle.py run  [--seeds 50 --trials 100]
    python experiment-sloped/run_oracle.py select [--top 5 --reps 8]
Outputs: experiment-sloped/results/{oracles.csv, selected_params.json}
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
SLOPE_START_Y = 2.0
CFG = of.sloped_cfg(slope_start_y=SLOPE_START_Y)
SCORE_FROM_Y = SLOPE_START_Y   # score from the slope crossing onward


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["run", "select", "all"])
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.stage in ("run", "all"):
        of.run_oracle("sloped", CFG, SCORE_FROM_Y, RESULTS,
                      a.seeds, a.trials, a.workers)
    if a.stage in ("select", "all"):
        of.select_oracle("sloped", CFG, SCORE_FROM_Y, RESULTS,
                         a.top, a.reps, a.workers)


if __name__ == "__main__":
    main()
