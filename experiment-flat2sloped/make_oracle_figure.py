"""Cross-terrain comparison of the per-terrain CPG oracle fits.

Reads the oracle distributions produced by experiment-flat and experiment-sloped
and produces:
  * figures/oracle_distribution.png   — one subfigure per CPG parameter, y = the
    optimal values, x = flat vs 10-deg sloped (distribution over BO seeds; black
    bar = median), showing the optimum is not unique and how it shifts;
  * results/selected_params.json      — the combined reference optima
    {flat, sloped} assembled from each terrain's selected_params.json, for the
    downstream event-trigger experiment and the problem-statement figures.

Usage (from repo root):
    python experiment-flat2sloped/make_oracle_figure.py
"""

import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from methods.cpg_bounds import bounds_lower, bounds_upper
from methods.oracle_fit import PARAM_LABELS
from methods.episode import PARAM_NAMES

FLAT_DIR = os.path.join(_REPO, "experiment-flat", "results")
SLOPE_DIR = os.path.join(_REPO, "experiment-sloped", "results")
FIG_OUT = os.path.join(_REPO, "figures", "oracle_distribution.png")
SELECTED_OUT = os.path.join(_HERE, "results", "selected_params.json")


def _load(results_dir):
    rows = list(csv.DictReader(open(os.path.join(results_dir, "oracles.csv"))))
    return np.array([[float(r[n]) for n in PARAM_NAMES] for r in rows])


def main():
    flat = _load(FLAT_DIR)
    slope = _load(SLOPE_DIR)
    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    C = {"flat": "#2a78d6", "sloped": "#eb6834"}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for j, ax in enumerate(axes.flat):
        for xi, (vals, key) in enumerate([(flat[:, j], "flat"),
                                          (slope[:, j], "sloped")]):
            jit = rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(np.full(len(vals), xi) + jit, vals, s=10,
                       color=C[key], alpha=0.45, lw=0)
            ax.hlines(np.median(vals), xi - 0.22, xi + 0.22,
                      color="#0b0b0b", lw=2, zorder=3)
        ax.set_title(PARAM_LABELS[j], fontsize=12)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["flat", r"$10^\circ$"], fontsize=10)
        pad = 0.04 * (hi[j] - lo[j])
        ax.set_ylim(lo[j] - pad, hi[j] + pad)
        ax.axhline(lo[j], color="#8a8984", lw=0.7, ls=":")
        ax.axhline(hi[j], color="#8a8984", lw=0.7, ls=":")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Optimal CPG parameters per terrain "
                 f"({len(flat)} / {len(slope)} independent BO fits, "
                 "30-s stability criterion $V$; black bar = median)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    print(f"saved {FIG_OUT}")

    # Combined reference optima for downstream use.
    flat_sel = json.load(open(os.path.join(FLAT_DIR, "selected_params.json")))
    slope_sel = json.load(open(os.path.join(SLOPE_DIR, "selected_params.json")))
    os.makedirs(os.path.dirname(SELECTED_OUT), exist_ok=True)
    with open(SELECTED_OUT, "w") as f:
        json.dump({"flat": flat_sel, "sloped": slope_sel}, f, indent=2)
    print(f"saved {SELECTED_OUT}")
    print(f"  flat   V={flat_sel['mean_V']:.3f} ({flat_sel['falls']} falls)")
    print(f"  sloped V={slope_sel['mean_V']:.3f} ({slope_sel['falls']} falls)")


if __name__ == "__main__":
    main()
