"""Fit CPG oracles on a 10-degree INCLINE.

Bayesian optimization over all 8 CPG parameters, each candidate scored by a 30-s
episode with the stability criterion V (methods.oracle_fit). Independent BO seeds
give a distribution of optima. On the incline the score is taken from the first
step past the slope start (`score_from_y`), so only walking ON the incline counts;
a candidate that never reaches the slope is scored as a failure.

The shared episode runner (methods.episode) feeds the trunk attitude back to the
CPG, so this fit is done with the VMC body-attitude feedback active (the
controller as actually used); disable it with JointCPG.ATTITUDE_FEEDBACK = False.

Usage (from repo root):
    python experiment-sloped/run_oracle.py            # run + select (10 seeds)
    python experiment-sloped/run_oracle.py run  [--seeds 10 --trials 60]
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
SLOPE_START_Y = 2.0                        # incline rises here
CFG = of.sloped_cfg(slope_deg=10.0, slope_start_y=SLOPE_START_Y)
SCORE_FROM_Y = SLOPE_START_Y               # score only once the robot is on the slope


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
        of.run_oracle("sloped", CFG, SCORE_FROM_Y, RESULTS,
                      a.seeds, a.trials, a.workers)
    if a.stage in ("select", "all"):
        of.select_oracle("sloped", CFG, SCORE_FROM_Y, RESULTS,
                         a.top, a.reps, a.workers)


if __name__ == "__main__":
    main()
