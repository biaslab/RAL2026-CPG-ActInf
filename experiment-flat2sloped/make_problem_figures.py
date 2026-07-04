"""Generate the Problem-Statement figures for root.tex.

Produces three files in ../figures/:
  terrain_flat.png / terrain_sloped.png  — Laikago walking on flat vs a 10 deg
      incline (PyBullet renders), illustrating the two terrains;
  optimal_params.png                      — the flat-optimal and slope-optimal
      8-D CPG parameter vectors, normalized to the shared bounds, showing that
      the optimum differs per terrain;
  noadapt_instability.png                 — trunk roll and body height after a
      flat->slope transition when the flat-optimal parameters are kept (robot
      destabilizes) versus switched to the slope-optimal parameters.

Usage (from repo root):
    python experiment-flat2sloped/make_problem_figures.py
"""

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

from methods import terrain
from methods.cpg_bounds import bounds_lower, bounds_upper
from methods.marxefe_optimizer import (get_base_orientation, load_environment,
                                       load_robot)
import run_experiment as fx   # experiment-flat2sloped/run_experiment.py

FIG_DIR = os.path.join(_REPO, "figures")
SEL = os.path.join(_HERE, "results", "selected_params.json")

# Consistent palette with the other paper figures.
C_FLAT = "#2a78d6"      # flat / keep flat-optimal
C_SLOPE = "#eb6834"     # sloped / switch to slope-optimal
C_TEXT = "#222222"
C_MUTED = "#8a8984"

PARAM_LABELS = [r"$\gamma$", r"$\omega_{\rm sw}$", r"$\omega_{\rm st}$",
                r"$F_{\rm fast}$", r"$K_{\rm stop}$", r"$A_{\rm hip}$",
                r"$A_{\rm knee}$", r"$b$"]

LEG_NAMES = ["FL", "FR", "RL", "RR"]


def load_params():
    with open(SEL) as f:
        s = json.load(f)
    return np.array(s["flat"]["params"]), np.array(s["sloped"]["params"])


# ── Figure 1: terrain renders ────────────────────────────────────────────────

def _render_walk(slope_start_y, params, capture_y, cam):
    """Walk the robot on the current terrain until its base passes capture_y,
    then return an RGB frame tracking it. `cam` sets the view."""
    import pybullet as p
    from methods.marxefe_optimizer import JointCPG

    terrain.TERRAIN_CONFIG = {"kind": "sloped", "slope_deg": fx.SLOPE_DEG,
                              "slope_start_y": float(slope_start_y),
                              "n_cols": fx.N_COLS}
    load_environment(fx.DT, use_gui=False)
    robot, _, joint_IDs_full, filtered, feet = load_robot(p)
    cpg = fx._reset_with_jitter(p, robot, seed=0)

    frame = None
    for k in range(int(18.0 / fx.DT)):
        raw = np.array([int(len(p.getContactPoints(
            bodyA=0, bodyB=robot, linkIndexA=-1, linkIndexB=feet[j])) > 0)
            for j in range(4)])
        hips, knees = cpg.step(params, raw, fx.DT)
        for j in range(4):
            a_id, h_id, k_id = joint_IDs_full[LEG_NAMES[j]]
            p.setJointMotorControl2(robot, a_id, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j])
            p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()
        base_pos, _ = get_base_orientation(p, robot, fx.DEFAULT_ORI)
        if base_pos[1] >= capture_y:
            gz = np.tan(np.deg2rad(fx.SLOPE_DEG)) * max(0.0, base_pos[1] - slope_start_y)
            tgt = [base_pos[0], base_pos[1], gz + 0.25]
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=tgt, distance=cam["d"], yaw=cam["yaw"],
                pitch=cam["pitch"], roll=0, upAxisIndex=2)
            proj = p.computeProjectionMatrixFOV(
                fov=cam["fov"], aspect=1.4, nearVal=0.05, farVal=30.0)
            _, _, rgb, _, _ = p.getCameraImage(
                980, 700, view, proj, renderer=p.ER_TINY_RENDERER,
                flags=p.ER_NO_SEGMENTATION_MASK,
                lightDirection=[-1.5, -2.0, 4.0])
            frame = np.reshape(rgb, (700, 980, 4))[:, :, :3].astype(np.uint8)
            break
    p.disconnect()
    return frame


def make_terrain_figs():
    flat_p, slope_p = load_params()
    # Side-on view so the incline profile is visible; walking is along +Y.
    cam = dict(d=2.3, yaw=88.0, pitch=-8.0, fov=45.0)

    flat = _render_walk(slope_start_y=fx.FAR_SLOPE_Y, params=flat_p,
                        capture_y=3.5, cam=cam)
    slope = _render_walk(slope_start_y=2.0, params=slope_p,
                         capture_y=4.0, cam=cam)

    for name, img, title in [("terrain_flat", flat, "flat terrain"),
                             ("terrain_sloped", slope, r"$10^\circ$ incline")]:
        fig, ax = plt.subplots(figsize=(3.4, 2.45))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, color=C_TEXT)
        fig.tight_layout(pad=0.1)
        out = os.path.join(FIG_DIR, name + ".png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


# ── Figure 2: optimal parameters per terrain ─────────────────────────────────

def make_param_fig():
    flat_p, slope_p = load_params()
    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    nf = (flat_p - lo) / (hi - lo)
    ns = (slope_p - lo) / (hi - lo)

    x = np.arange(8)
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.bar(x - w / 2, nf, w, color=C_FLAT, label="flat optimum")
    ax.bar(x + w / 2, ns, w, color=C_SLOPE, label=r"$10^\circ$ incline optimum")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_LABELS, fontsize=11)
    ax.set_ylabel("normalized value", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.text(-0.6, -0.16, "lower bound", fontsize=7, color=C_MUTED, ha="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "optimal_params.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ── Figure 3: not adapting -> instability ────────────────────────────────────

FALL_TILT_DEG = np.rad2deg(np.arccos(0.3))   # fall threshold (magic_value < 0.3)


def _episode_log(seed, slope_start_y, params_start, params_target=None,
                 switch_step=None, duration=fx.T_TOTAL):
    """Run one episode and log forward position and the trunk's tilt from
    vertical (the convention-free attitude measure the fall test uses:
    cos(tilt) = world_up . body_up). Returns t, y, tilt_deg, fell, fall_step."""
    import pybullet as p
    from methods.marxefe_optimizer import check_if_fallen, get_base_orientation

    terrain.TERRAIN_CONFIG = {"kind": "sloped", "slope_deg": fx.SLOPE_DEG,
                              "slope_start_y": float(slope_start_y),
                              "n_cols": fx.N_COLS}
    load_environment(fx.DT, use_gui=False)
    robot, _, joint_IDs_full, filtered, feet = load_robot(p)
    cpg = fx._reset_with_jitter(p, robot, seed)

    n = int(round(duration / fx.DT))
    t = np.arange(n) * fx.DT
    y = np.zeros(n)
    tilt = np.zeros(n)
    fell, fall_step = False, None
    for k in range(n):
        if params_target is not None and switch_step is not None and k >= switch_step:
            frac = min(1.0, (k - switch_step) / fx.SWITCH_RAMP_STEPS)
            applied = params_start + frac * (params_target - params_start)
        else:
            applied = params_start
        raw = np.array([int(len(p.getContactPoints(
            bodyA=0, bodyB=robot, linkIndexA=-1, linkIndexB=feet[j])) > 0)
            for j in range(4)])
        hips, knees = cpg.step(applied, raw, fx.DT)
        for j in range(4):
            a_id, h_id, k_id = joint_IDs_full[LEG_NAMES[j]]
            p.setJointMotorControl2(robot, a_id, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j])
            p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()
        base_pos, base_ori = get_base_orientation(p, robot, fx.DEFAULT_ORI)
        up = p.getMatrixFromQuaternion(base_ori)[6:]
        y[k] = base_pos[1]
        tilt[k] = np.rad2deg(np.arccos(np.clip(np.dot([0, 0, 1], up), -1, 1)))
        is_fallen, _, _, _ = check_if_fallen(p, robot, base_pos, base_ori)
        if is_fallen:
            fell, fall_step = True, k
            t, y, tilt = t[:k + 1], y[:k + 1], tilt[:k + 1]
            break
    p.disconnect()
    return dict(t=t, y=y, tilt=tilt, fell=fell, fall_step=fall_step)


def make_instability_fig():
    flat_p, slope_p = load_params()
    k_switch = int(round(fx.T_SWITCH / fx.DT))

    # Find a seed where keeping flat-optimal falls but switching stays upright,
    # for a clean and representative contrast.
    chosen = None
    for seed in range(40):
        cal = _episode_log(seed, fx.FAR_SLOPE_Y, flat_p, duration=fx.T_SWITCH)
        if cal["fell"]:
            continue
        y10 = float(cal["y"][-1])
        keep = _episode_log(seed, y10, flat_p)
        if not (keep["fell"] and keep["fall_step"] > k_switch):
            continue
        switch = _episode_log(seed, y10, flat_p, params_target=slope_p,
                              switch_step=k_switch)
        if switch["fell"]:
            continue
        chosen = (seed, y10, keep, switch)
        print(f"instability example: seed {seed}, y10={y10:.2f}")
        break
    if chosen is None:
        raise RuntimeError("no clean keep-falls / switch-survives seed found")

    seed, y10, keep, switch = chosen
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)

    def tt(log):
        return log["t"] - fx.T_SWITCH

    series = [(keep, C_FLAT, "keep flat-optimal"),
              (switch, C_SLOPE, "switch to incline-optimal")]

    ax = axes[0]
    for log, c, lab in series:
        ax.plot(tt(log), log["tilt"], color=c, lw=1.6, label=lab)
        if log["fell"]:
            ax.plot(tt(log)[-1], log["tilt"][-1], "x", color=c, ms=8, mew=2)
    ax.axhline(FALL_TILT_DEG, color=C_MUTED, lw=0.9, ls=":")
    ax.text(-9.5, FALL_TILT_DEG - 2, "fall threshold", fontsize=7.5,
            color=C_MUTED, va="top")
    ax.set_ylabel("trunk tilt from vertical [deg]", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="center left")

    ax = axes[1]
    for log, c, lab in series:
        ax.plot(tt(log), log["y"] - y10, color=c, lw=1.6)
        if log["fell"]:
            ax.plot(tt(log)[-1], log["y"][-1] - y10, "x", color=c, ms=8, mew=2)
    ax.set_ylabel("distance up the incline [m]", fontsize=9)
    ax.annotate("fall", xy=(tt(keep)[-1], keep["y"][-1] - y10), fontsize=8,
                color=C_FLAT, xytext=(-24, 2), textcoords="offset points")

    for ax in axes:
        ax.axvline(0, color=C_MUTED, lw=1, ls="--")
        ax.text(0.2, ax.get_ylim()[1], "terrain\nchange", fontsize=7.5,
                color=C_MUTED, va="top", ha="left")
        ax.set_xlabel("time since terrain change [s]", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "noadapt_instability.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    make_param_fig()
    make_instability_fig()
    make_terrain_figs()
