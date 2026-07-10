"""Generate the PROBLEM STATEMENT figures for root.tex, using the same episode
dynamics as the problem-statement notebook (methods.episode.run_episode).

Produces, in ../figures/:
  * optimal_params.{pdf,png}     -- flat vs 10-deg optimal CPG parameters
                                    (the differing ones), normalised to bounds.
  * noadapt_instability.{pdf,png}-- one representative bout: holding the flat
                                    optimum vs switching to the incline optimum
                                    at the transition. Left: trunk tilt from
                                    vertical; right: forward progress.
  * terrain_flat.png / terrain_sloped.png (best-effort PyBullet renders).

Run from repo root:  python problem-statement/make_paper_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from methods.episode import DT, N_COLS, run_episode                     # noqa: E402
from methods.cpg_bounds import bounds_lower, bounds_upper               # noqa: E402

FIG_DIR = os.path.join(_REPO, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SLOPE_DEG = 10.0          # paper's incline
SLOPE_START_Y = 5.0       # the robot meets the incline 5 m in (as in the notebook)
DURATION = 30.0           # bout length [s]
FALL_TILT_DEG = np.rad2deg(np.arccos(0.3))   # upright<0.3 fall criterion -> ~72.5 deg

PARAM_LABELS = [r"$\gamma$", r"$\omega_{\rm sw}$", r"$\omega_{\rm st}$",
                r"$F_{\rm fast}$", r"$K_{\rm stop}$", r"$A_{\rm hip}$",
                r"$A_{\rm knee}$", r"$b$"]

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 150})
C_KEEP, C_SWITCH = "#d1495b", "#2a9d8f"       # keep=red, switch=teal


def load_params():
    with open(os.path.join(_REPO, "experiment-flat", "results",
                           "selected_params.json")) as f:
        flat = np.array(json.load(f)["params"], float)
    with open(os.path.join(_REPO, "experiment-sloped", "results",
                           "selected_params.json")) as f:
        slope = np.array(json.load(f)["params"], float)
    return flat, slope


def tilt_from_vertical(roll, pitch):
    """Total trunk tilt from vertical [deg] from roll and pitch (physical
    convention); cos(tilt) = cos(roll) cos(pitch)."""
    return np.rad2deg(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1.0, 1.0)))


# ── Figure 1: terrain-dependent optimal parameters ───────────────────────────

def fig_optimal_params(flat, slope):
    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    fn = (flat - lo) / (hi - lo)
    sn = (slope - lo) / (hi - lo)
    diff = np.abs(flat - slope) > 1e-6                 # parameters that differ
    idx = np.where(diff)[0]
    x = np.arange(len(idx)); w = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.bar(x - w/2, fn[idx], w, label="flat optimum",  color="#4c72b0")
    ax.bar(x + w/2, sn[idx], w, label="incline optimum", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels([PARAM_LABELS[i] for i in idx])
    ax.set_ylabel("value (normalised\nto parameter bounds)")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"Terrain-conditioned CPG optima $\theta^\star(e)$")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.32))
    ax.grid(axis="y", alpha=0.3); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"optimal_params.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"optimal_params: differing params {[PARAM_LABELS[i] for i in idx]}")


# ── Figure 2: hold vs switch representative bout ─────────────────────────────

def _cfg():
    return {"kind": "sloped", "slope_deg": SLOPE_DEG,
            "slope_start_y": SLOPE_START_Y, "n_cols": N_COLS}


def _run_pair(seed, flat, slope):
    """Hold-flat and switch-at-transition bouts sharing an identical prefix."""
    hold = run_episode(_cfg(), seed, flat, duration=DURATION)
    onset = np.nonzero(np.asarray(hold["y"]) >= SLOPE_START_Y)[0]
    k_sw = int(onset[0]) if len(onset) else None
    if k_sw is None:
        return hold, None, None
    switch = run_episode(_cfg(), seed, flat, params_target=slope,
                         switch_step=k_sw, duration=DURATION)
    return hold, switch, k_sw


def _score(hold, switch, k_sw):
    """Prefer a seed where holding falls and switching survives longer."""
    if switch is None:
        return -1e9
    hf = bool(hold["fell"]); sf = bool(switch["fell"])
    h_end = hold["fall_step"] if hf else len(hold["y"])
    s_end = switch["fall_step"] if sf else len(switch["y"])
    return (2.0 if (hf and not sf) else 0.0) + (s_end - h_end) * DT * 0.1


def fig_noadapt(flat, slope, seeds=range(12)):
    best = None
    for s in seeds:
        hold, switch, k_sw = _run_pair(s, flat, slope)
        if k_sw is None:
            continue
        sc = _score(hold, switch, k_sw)
        hf = bool(hold["fell"]); sf = bool(switch["fell"])
        print(f"  seed {s:2d}: onset t={k_sw*DT:4.1f}s  hold {'FELL@%.1f'%(hold['fall_step']*DT) if hf else 'ok  '}"
              f"  switch {'FELL@%.1f'%(switch['fall_step']*DT) if sf else 'ok  '}  score={sc:.2f}")
        if best is None or sc > best[0]:
            best = (sc, s, hold, switch, k_sw)
    sc, seed, hold, switch, k_sw = best
    t_sw = k_sw * DT
    print(f"-> representative seed {seed} (score {sc:.2f}), transition t={t_sw:.1f}s")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for lab, log, col in [("keep flat optimum", hold, C_KEEP),
                          ("switch to incline optimum", switch, C_SWITCH)]:
        t = np.asarray(log["t"]) if "t" in log else np.arange(len(log["y"])) * DT
        tilt = tilt_from_vertical(np.asarray(log["roll"]), np.asarray(log["pitch"]))
        axL.plot(t, tilt, color=col, lw=1.8, label=lab)
        axR.plot(t, np.asarray(log["y"]) - SLOPE_START_Y, color=col, lw=1.8, label=lab)
        if log["fell"]:
            tf = log["fall_step"] * DT
            axL.plot(tf, tilt[-1], "x", color=col, ms=9, mew=2)
            axR.plot(tf, log["y"][-1] - SLOPE_START_Y, "x", color=col, ms=9, mew=2)

    axL.axhline(FALL_TILT_DEG, color="0.5", ls="--", lw=1.0)
    axL.text(DURATION * 0.98, FALL_TILT_DEG + 1.5, "fall threshold",
             fontsize=8, color="0.4", ha="right", va="bottom")
    for ax in (axL, axR):
        ax.axvline(t_sw, color="0.35", ls=":", lw=1.2)
        ax.text(t_sw + 0.3, ax.get_ylim()[1] * 0.5, "terrain change", rotation=90,
                fontsize=8, color="0.35", va="center")
    axL.set_xlabel("time [s]"); axL.set_ylabel("trunk tilt from vertical [deg]")
    axR.set_xlabel("time [s]"); axR.set_ylabel("distance onto incline [m]")
    axL.set_title("Attitude"); axR.set_title("Forward progress")
    axL.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"noadapt_instability.{ext}"), bbox_inches="tight")
    plt.close(fig)
    return seed, t_sw


# ── Best-effort robot renders ────────────────────────────────────────────────

def render_terrains():
    try:
        import pybullet as p
        from methods import terrain
        from methods.marxefe_optimizer import load_robot
        for name, cfg in [("terrain_flat",
                           {"kind": "sloped", "slope_deg": SLOPE_DEG,
                            "slope_start_y": 60.0, "n_cols": N_COLS}),
                          ("terrain_sloped",
                           {"kind": "sloped", "slope_deg": SLOPE_DEG,
                            "slope_start_y": 0.5, "n_cols": N_COLS})]:
            terrain.TERRAIN_CONFIG = dict(cfg)
            p.connect(p.DIRECT)
            import pybullet_data
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.8)
            terrain.build_ground(p)
            load_robot(p)
            for _ in range(50):
                p.stepSimulation()
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 1.0, 0.4], distance=2.6,
                yaw=50, pitch=-20, roll=0, upAxisIndex=2)
            proj = p.computeProjectionMatrixFOV(fov=55, aspect=1.3, nearVal=0.1, farVal=20)
            w, h, rgb, _, _ = p.getCameraImage(520, 400, view, proj,
                                               renderer=p.ER_TINY_RENDERER)
            img = np.reshape(rgb, (h, w, 4))[:, :, :3]
            plt.imsave(os.path.join(FIG_DIR, f"{name}.png"), img.astype(np.uint8))
            p.disconnect()
            print(f"rendered {name}.png")
        return True
    except Exception as e:
        print(f"[render] skipped robot renders ({e})")
        try:
            import pybullet as p
            p.disconnect()
        except Exception:
            pass
        return False


if __name__ == "__main__":
    flat, slope = load_params()
    print("flat  optimum:", np.round(flat, 3).tolist())
    print("slope optimum:", np.round(slope, 3).tolist())
    fig_optimal_params(flat, slope)
    print("selecting representative seed for noadapt figure:")
    seed, t_sw = fig_noadapt(flat, slope)
    render_terrains()
    print(f"\nsaved figures to {FIG_DIR}")
