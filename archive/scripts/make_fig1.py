"""Generate Fig. 1: Terrain change creates adaptation urgency.

Panel (a) — Schematic performance landscape J(θ) for two terrain conditions,
illustrating the shift in optimal CPG parameters after a terrain change.

Panel (b) — Schematic locomotion quality over sequential trials after a terrain
change, comparing fast adaptation (proposed), slow adaptation (BO), and no
adaptation (grid search). The fall threshold and adaptation deadline T_dead
are marked to motivate rapid continual learning.

Run from the repository root:
    python make_fig1.py
Output: figures/fig1_adaptation_urgency.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Plot style — close to IEEE paper body font
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        8,
    "axes.labelsize":   8,
    "axes.titlesize":   8,
    "legend.fontsize":  7,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "axes.linewidth":   0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth":  1.4,
})

# Colour palette (colour-blind friendly, also readable in grey-scale via line style)
C_FLAT   = "#2166ac"   # blue  — flat terrain
C_NEW    = "#d6604d"   # red   — new terrain (friction / slope)
C_FAST   = "#1a9641"   # green — fast adaptation (proposed)
C_SLOW   = "#4393c3"   # blue  — slow adaptation (BO)
C_NONE   = "#808080"   # grey  — no adaptation (grid search)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.16, 2.6),      # 7.16 in = IEEE full-text-width
    gridspec_kw={"wspace": 0.38},
)

# ===========================================================================
# Panel (a) — Shifting performance landscape
# ===========================================================================
theta = np.linspace(0, 10, 400)

# Schematic Gaussian performance landscapes (normalised, y in [0,1])
mu_flat, mu_new = 3.5, 6.2
sig = 1.8
J_flat = np.exp(-0.5 * ((theta - mu_flat) / sig) ** 2)
J_new  = 0.88 * np.exp(-0.5 * ((theta - mu_new)  / sig) ** 2)

ax1.plot(theta, J_flat, color=C_FLAT, ls="-",  lw=1.5,
         label=r"$J(\theta,\tau)$ — original terrain")
ax1.plot(theta, J_new,  color=C_NEW,  ls="--", lw=1.5,
         label=r"$J(\theta,\tau')$ — new terrain")

# Mark the two optima
J_flat_peak = J_flat[np.argmin(np.abs(theta - mu_flat))]
J_new_peak  = J_new [np.argmin(np.abs(theta - mu_new ))]
ax1.scatter([mu_flat], [J_flat_peak], color=C_FLAT, s=28, zorder=5)
ax1.scatter([mu_new ], [J_new_peak ], color=C_NEW,  s=28, zorder=5, marker="D")

ax1.annotate(r"$\theta^*(\tau)$",
             xy=(mu_flat, J_flat_peak), xytext=(mu_flat - 2.6, J_flat_peak + 0.05),
             fontsize=7.5, color=C_FLAT,
             arrowprops=dict(arrowstyle="-|>", color=C_FLAT, lw=0.8))
ax1.annotate(r"$\theta^*(\tau')$",
             xy=(mu_new, J_new_peak), xytext=(mu_new + 0.3, J_new_peak + 0.08),
             fontsize=7.5, color=C_NEW,
             arrowprops=dict(arrowstyle="-|>", color=C_NEW, lw=0.8))

# Show the robot's current position on the new-terrain landscape right after change
J_at_old_on_new = 0.88 * np.exp(-0.5 * ((mu_flat - mu_new) / sig) ** 2)
ax1.scatter([mu_flat], [J_at_old_on_new], color="k", s=28, zorder=6, marker="x")
ax1.annotate(r"$\theta_k$ (after $\tau\!\to\!\tau'$)",
             xy=(mu_flat, J_at_old_on_new),
             xytext=(mu_flat - 3.1, J_at_old_on_new - 0.18),
             fontsize=7, color="k",
             arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8))

# Mismatch arrow δ
y_arrow = 0.26
ax1.annotate("", xy=(mu_new, y_arrow), xytext=(mu_flat, y_arrow),
             arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
ax1.text((mu_flat + mu_new) / 2, y_arrow + 0.03, r"$\delta_k$",
         ha="center", va="bottom", fontsize=8)

ax1.set_xlabel(r"CPG parameter $\theta$ (representative axis)", labelpad=2)
ax1.set_ylabel(r"Locomotion quality $J(\theta, \tau)$")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1.22)
ax1.set_xticks([])
ax1.set_yticks([0, 0.5, 1.0])
ax1.yaxis.set_tick_params(labelsize=6.5)
ax1.legend(loc="upper left", framealpha=0.85, handlelength=1.6, borderpad=0.5)
ax1.set_title("(a) Terrain change shifts the performance landscape",
              pad=4, fontsize=8)

# ===========================================================================
# Panel (b) — Adaptation deadline
# ===========================================================================
trials = np.arange(0, 16)   # trial index; 0 = last trial before terrain change

# Schematic trajectories (normalised so that pre-change optimum = 1.0)
# Fast adaptation (proposed MARXEFE-style):
# quick recovery within ~3 trials
J_fast = 1.0 - 0.85 * np.exp(-1.1 * trials) * (1 - 0.1 * np.random.default_rng(0).standard_normal(len(trials)) * 0)
J_fast[0] = 1.0  # pre-change performance
J_fast[1] = 0.22 # terrain-change impact
J_fast[2:] = 0.22 + 0.72 * (1 - np.exp(-0.9 * (trials[2:] - 1)))

# Slow adaptation (BO-style): recovers but after many trials
J_slow = np.zeros_like(trials, dtype=float)
J_slow[0] = 1.0
J_slow[1] = 0.15
J_slow[2:] = 0.15 + 0.62 * (1 - np.exp(-0.28 * (trials[2:] - 1)))

# No adaptation (grid search): fixed parameters, slow passive drift downward
J_none = np.zeros_like(trials, dtype=float)
J_none[0] = 1.0
J_none[1:] = 0.18 - 0.04 * trials[1:]

# Fall threshold
J_fall = 0.0

# Plot trajectories
ax2.plot(trials, J_fast, color=C_FAST, ls="-",  lw=1.5,
         label="Fast adaptation (proposed)")
ax2.plot(trials, J_slow, color=C_SLOW, ls="--", lw=1.5,
         label="Slow adaptation (BO)")
ax2.plot(trials, J_none, color=C_NONE, ls=":",  lw=1.5,
         label="No adaptation (grid search)")

# Fall threshold line
ax2.axhline(J_fall, color="k", lw=0.9, ls="-.", alpha=0.7, label="Fall threshold")

# Shade the danger zone
ax2.fill_between(trials, J_fall - 0.35, J_fall, color="salmon", alpha=0.18,
                 zorder=0)

# Mark terrain-change event
ax2.axvline(1, color="k", lw=0.8, ls="-", alpha=0.45)
ax2.text(1.15, 0.98, r"$\tau\!\to\!\tau'$", va="top", fontsize=7, color="k",
         alpha=0.7)

# Mark adaptation deadline T_dead (where no-adaptation crosses the threshold)
# J_none[1:] = 0.18 - 0.04*t => crosses 0 at t = 0.18/0.04 = 4.5 → trial 5.5
T_dead = 1 + int(0.18 / 0.04)   # ~5th trial after change
ax2.axvline(1 + T_dead - 1, color="firebrick", lw=0.9, ls="--", alpha=0.7)
ax2.text(1 + T_dead - 1 + 0.15, -0.28, r"$T_{\mathrm{dead}}$",
         fontsize=7.5, color="firebrick", va="bottom")

# Fall markers for slow and no-adaptation at T_dead
for J_traj, col in [(J_none, C_NONE), (J_slow, C_SLOW)]:
    idx = np.searchsorted(trials, 1 + T_dead - 1)
    j_val = J_traj[min(idx, len(J_traj)-1)]
    if j_val < J_fall:
        ax2.scatter([1 + T_dead - 1], [j_val], color=col, s=60, marker="X",
                    zorder=6)

ax2.set_xlabel("Trial number after terrain change", labelpad=2)
ax2.set_ylabel(r"Locomotion quality $J(\theta_k, \tau')$")
ax2.set_xlim(0, 15)
ax2.set_ylim(-0.45, 1.18)
ax2.set_xticks([0, 1, 5, 10, 15])
ax2.set_xticklabels([r"$-1$", "0", "4", "9", "14"])
ax2.legend(loc="lower right", framealpha=0.85, handlelength=1.6, borderpad=0.5)
ax2.set_title("(b) Adaptation speed determines recovery before $T_{\\mathrm{dead}}$",
              pad=4, fontsize=8)
ax2.grid(True, axis="y", alpha=0.22, lw=0.6)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig1_adaptation_urgency.pdf")
fig.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"Saved {out_path}")
