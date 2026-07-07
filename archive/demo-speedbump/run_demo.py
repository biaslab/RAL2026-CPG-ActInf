"""Speed-bump adaptation demo — dashboard video.

The Laikago walks on flat ground with the flat-terrain-optimal CPG parameters
while a MARX model (the MARX-EFE agent's generative model, used purely as a
predictor — no EFE control) tracks the base dynamics. When the robot hits a
speed bump (slope up, flat top, slope down — a `multislope` terrain), the
0.5-s-ahead rollout prediction error spikes; the demo reacts by ramping the
CPG parameters over to the sloped-terrain optimum.

Output: a 1280x720 MP4 dashboard —
  left   : zoomed-in tracking render of the robot (full height),
  right  : (top) running prediction error, (bottom) 8 slider widgets
           showing the live CPG parameter vector.

Usage (from the repo root or this folder):
  python demo-speedbump/run_demo.py            # simulate + compose video
  python demo-speedbump/run_demo.py --sim-only
  python demo-speedbump/run_demo.py --compose-only   # reuse saved sim data
"""

import argparse
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

import pybullet as p                                    # noqa: E402
from methods import terrain                             # noqa: E402
from methods.cpg_bounds import bounds_lower, bounds_upper   # noqa: E402
from methods.marxefe_optimizer import (                 # noqa: E402
    build_marx_agent, check_if_fallen, extract_observation,
    load_environment, load_robot, reset_simulation)

# ── Configuration ────────────────────────────────────────────────────────────
DT             = 0.01          # simulation / control timestep [s]
RUN_TIME       = 12.0          # episode length [s]
RENDER_EVERY   = 3             # render 1 frame per 3 control steps -> 33.3 fps
IMG_W, IMG_H   = 620, 720      # left-panel render size (full video height)
VIDEO_W, VIDEO_H = 1280, 720

# Speed bump: flat -> up-slope -> flat top -> down-slope -> flat again.
BUMP_START = 1.8               # forward position where the up-slope begins [m]
BUMP_SLOPE = 10.0              # ramp angle [deg]
UP_LEN     = 1.0               # up-ramp forward length [m]
TOP_LEN    = 0.7               # flat-top forward length [m]
BUMP_END   = BUMP_START + UP_LEN + TOP_LEN + UP_LEN     # back at ground level

# Optimized CPG parameter vectors from figures/cpg_optima_by_parameter.csv
# (seed 0 rows). Order: [coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN,
#                        hip_amp, knee_amp, b]
PARAMS_FLAT   = np.array([4.0, 10.0, 25.0, 60.0, 0.316, 0.1, 0.729, 7.762])
PARAMS_SLOPED = np.array([4.0, 10.0, 25.0, 25.0, 0.423, 0.1, 0.801, 10.0])

PARAM_NAMES = ["coupling_gain", "w_swing", "w_stance", "F_FAST",
               "STOP_GAIN", "hip_amp", "knee_amp", "b"]

# Prediction-error monitor. The signal is the H_PRED-step-ahead rollout error:
# at every step the MARX model predicts the observation 0.5 s into the future
# (assuming the controls stay constant); when reality catches up, the norm of
# the difference is the prediction error. One-step innovations at dt=10 ms are
# dominated by persistence and barely register a terrain change; the multi-step
# rollout compounds the model mismatch and spikes visibly on the bump.
# The EMA of the signal is compared against a baseline collected on flat
# ground; the switch fires when it exceeds baseline_mean + K_SIGMA*baseline_std.
H_PRED         = 50            # prediction horizon [steps] (0.5 s)
WARMUP_UPDATES = 150           # model updates before rollout predictions start
EMA_ALPHA      = 0.08
BASELINE_T     = (2.4, 3.4)    # time window (s) that defines "flat" baseline
ARM_TIME       = 3.5           # do not allow a switch before this time [s]
K_SIGMA        = 6.0
SWITCH_RAMP    = 40            # steps over which new params are ramped in

DATA_PATH  = os.path.join(_HERE, "demo_data.pkl")
FRAMES_DIR = os.path.join(_HERE, "frames")
VIDEO_PATH = os.path.join(_HERE, "speedbump_dashboard.mp4")

CAMERA = dict(distance=1.45, yaw=52.0, pitch=-12.0, fov=55.0, z_offset=-0.25)


# ── Simulation ───────────────────────────────────────────────────────────────

def render_frame(cam_target):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(cam_target), distance=CAMERA["distance"],
        yaw=CAMERA["yaw"], pitch=CAMERA["pitch"], roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(
        fov=CAMERA["fov"], aspect=IMG_W / IMG_H, nearVal=0.05, farVal=25.0)
    _, _, rgb, _, _ = p.getCameraImage(
        IMG_W, IMG_H, view, proj, renderer=p.ER_TINY_RENDERER,
        flags=p.ER_NO_SEGMENTATION_MASK,
        lightDirection=[-2.0, 2.0, 4.0])
    return np.reshape(rgb, (IMG_H, IMG_W, 4))[:, :, :3].astype(np.uint8)


def add_bump_markers():
    """Thin colored strips on the terrain marking the bump's footprint (visual
    only, collision-free; created after the ground so body 0 stays the ground)."""
    tan = np.tan(np.deg2rad(BUMP_SLOPE))
    top_h = UP_LEN * tan

    def strip(y, z, rgba):
        vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.5, 0.02, 0.004],
                                 rgbaColor=rgba)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs,
                          basePosition=[0, y, z + 0.004])

    amber = [0.93, 0.63, 0.0, 1.0]
    strip(BUMP_START, 0.0, amber)
    strip(BUMP_START + UP_LEN, top_h, amber)
    strip(BUMP_START + UP_LEN + TOP_LEN, top_h, amber)
    strip(BUMP_END, 0.0, amber)


def simulate(save_frames=True):
    terrain.TERRAIN_CONFIG = {
        "kind": "multislope",
        "segments": [(BUMP_START, BUMP_SLOPE),
                     (BUMP_START + UP_LEN, 0.0),
                     (BUMP_START + UP_LEN + TOP_LEN, -BUMP_SLOPE),
                     (BUMP_END, 0.0)],
    }
    load_environment(DT, use_gui=False)
    robot, _, joint_IDs_full, filtered_joint_IDs, feet_joint_IDs = load_robot(p)
    add_bump_markers()

    ori_default = [0.0, 0.5, 0.5, 0.0]
    cpg = reset_simulation(p, robot, filtered_joint_IDs, ori_default)

    # MARX model used purely as a one-step-ahead predictor (no EFE control).
    np.random.seed(0)
    agent = build_marx_agent(target_velocity=1.0, forgetting=0.995)

    y_k, base_pos, base_orientation = extract_observation(p, robot, ori_default)

    num_steps = int(RUN_TIME / DT)
    leg_names = ["FL", "FR", "RL", "RR"]

    times     = np.zeros(num_steps)
    err_raw   = np.zeros(num_steps)
    err_h     = np.zeros(num_steps)
    err_ema   = np.zeros(num_steps)
    pred_multi = np.full((num_steps + H_PRED, 4), np.nan)
    params_l  = np.zeros((num_steps, 8))
    pos_fwd   = np.zeros(num_steps)
    pos_z     = np.zeros(num_steps)
    mode      = np.zeros(num_steps, dtype=int)   # 0 = flat set, 1 = sloped set

    frame_steps = []
    switched     = False
    switch_step  = None
    seg_start    = PARAMS_FLAT.copy()
    seg_target   = PARAMS_FLAT.copy()
    seg_anchor   = 0
    ema          = 0.0

    cam_target = np.array([0.0, 0.0, 0.35])

    if save_frames:
        os.makedirs(FRAMES_DIR, exist_ok=True)
        import imageio.v2 as imageio

    actual_steps = num_steps
    fell = False
    for k in range(num_steps):
        t = k * DT
        times[k] = t

        # Applied parameters: hold the current set; ramp toward the sloped set
        # over SWITCH_RAMP steps once the monitor has fired.
        frac = min(1.0, (k - seg_anchor) / max(1, SWITCH_RAMP)) if switched else 1.0
        applied = seg_start + frac * (seg_target - seg_start) if switched else seg_target
        applied = np.clip(applied, bounds_lower.numpy(), bounds_upper.numpy())
        params_l[k] = applied
        mode[k] = int(switched)

        # CPG step -> joint targets (identical to the validated episode loop).
        raw_contacts = np.array([
            int(len(p.getContactPoints(bodyA=0, bodyB=robot, linkIndexA=-1,
                                       linkIndexB=feet_joint_IDs[j])) > 0)
            for j in range(4)])
        hip_angles, knee_angles = cpg.step(applied, raw_contacts, DT)
        for j in range(4):
            abd, hip, knee = joint_IDs_full[leg_names[j]]
            p.setJointMotorControl2(robot, abd, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, hip, p.POSITION_CONTROL, hip_angles[j])
            p.setJointMotorControl2(robot, knee, p.POSITION_CONTROL, knee_angles[j])
        p.stepSimulation()

        y_new, base_pos, base_orientation = extract_observation(p, robot, ori_default)
        pos_fwd[k] = base_pos[1]
        pos_z[k]   = base_pos[2]

        # One-step prediction error: predict y_new from the regressor the MARX
        # update will use (current control included), then assimilate.
        ub = agent.backshift(agent.ubuffer, applied)
        x_k = np.concatenate([ub.flatten(), agent.ybuffer.flatten()])
        _, mu_k, _ = agent.posterior_predictive(x_k)
        err_raw[k] = float(np.linalg.norm(y_new - mu_k))   # one-step innovation
        agent.update(y_new, applied)

        # H_PRED-step rollout error: compare y at step k with the prediction
        # issued H_PRED steps ago; then issue a new prediction for step k+H_PRED
        # (assuming controls stay at their current value).
        if np.isfinite(pred_multi[k]).all():
            e = float(np.linalg.norm(y_new - pred_multi[k]))
        else:
            e = 0.0
        if agent.n_updates > WARMUP_UPDATES:
            m_pred, _ = agent.predictions(
                np.tile(applied[:, None], (1, H_PRED)), time_horizon=H_PRED)
            if k + H_PRED < pred_multi.shape[0]:
                pred_multi[k + H_PRED] = m_pred[:, -1]

        err_h[k] = e
        ema = e if k == 0 else (1 - EMA_ALPHA) * ema + EMA_ALPHA * e
        err_ema[k] = ema

        # Switch monitor: baseline stats from the flat-ground window, fire once.
        if not switched and t >= ARM_TIME:
            b0, b1 = int(BASELINE_T[0] / DT), int(BASELINE_T[1] / DT)
            base_mu, base_sd = err_ema[b0:b1].mean(), err_ema[b0:b1].std()
            if ema > base_mu + K_SIGMA * base_sd:
                switched   = True
                switch_step = k
                seg_start  = applied.copy()
                seg_target = PARAMS_SLOPED.copy()
                seg_anchor = k
                print(f"[demo] prediction-error spike at t={t:.2f}s "
                      f"(y={base_pos[1]:.2f} m): switching to sloped params")

        # Camera: low-pass-filtered target so the view tracks without jitter.
        target_now = np.array([base_pos[0], base_pos[1],
                               base_pos[2] + CAMERA["z_offset"]])
        cam_target = 0.92 * cam_target + 0.08 * target_now

        if save_frames and (k % RENDER_EVERY == 0):
            frame = render_frame(cam_target)
            imageio.imwrite(os.path.join(FRAMES_DIR, f"f{len(frame_steps):05d}.jpg"),
                            frame, quality=92)
            frame_steps.append(k)

        y_k = y_new
        fell, _, _, _ = check_if_fallen(p, robot, base_pos, base_orientation)
        if fell:
            print(f"[demo] robot fell at t={t:.2f}s (y={base_pos[1]:.2f} m)")
            actual_steps = k + 1
            break

    p.disconnect()

    data = {
        "dt": DT, "times": times[:actual_steps], "err_raw": err_raw[:actual_steps],
        "err_h": err_h[:actual_steps],
        "err_ema": err_ema[:actual_steps], "params": params_l[:actual_steps],
        "pos_fwd": pos_fwd[:actual_steps], "pos_z": pos_z[:actual_steps],
        "mode": mode[:actual_steps], "switch_step": switch_step,
        "frame_steps": frame_steps, "fell": fell,
        "bump": (BUMP_START, BUMP_END), "switch_ramp": SWITCH_RAMP,
        "params_flat": PARAMS_FLAT, "params_sloped": PARAMS_SLOPED,
    }
    with open(DATA_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"[demo] saved {actual_steps} steps, {len(frame_steps)} frames, "
          f"final y={pos_fwd[actual_steps-1]:.2f} m, fell={fell}")
    return data


# ── Dashboard video composition ──────────────────────────────────────────────

# Palette (dataviz reference palette, light mode).
C_SURFACE = "#fcfcfb"
C_TEXT    = "#0b0b0b"
C_TEXT2   = "#52514e"
C_MUTED   = "#8a8984"
C_BLUE    = "#2a78d6"       # series 1: prediction error / slider knobs
C_ORANGE  = "#eb6834"       # switch event / sloped regime accent
C_TRACK   = "#e4e3df"
C_BAND    = "#efeeea"


def compose():
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(DATA_PATH, "rb") as f:
        d = pickle.load(f)

    times, err_raw, err_ema = d["times"], d["err_h"], d["err_ema"]
    params, frame_steps = d["params"], d["frame_steps"]
    switch_step = d["switch_step"]
    t_switch = times[switch_step] if switch_step is not None else None

    # Times at which the robot's base is over the bump footprint.
    on_bump = (d["pos_fwd"] >= d["bump"][0]) & (d["pos_fwd"] <= d["bump"][1])
    t_enter = times[on_bump][0] if on_bump.any() else None
    t_exit  = times[on_bump][-1] if on_bump.any() else None

    lo, hi = bounds_lower.numpy(), bounds_upper.numpy()
    norm = lambda v: (v - lo) / (hi - lo)

    dpi = 100
    fig = plt.figure(figsize=(VIDEO_W / dpi, VIDEO_H / dpi), dpi=dpi)
    fig.patch.set_facecolor(C_SURFACE)

    # Left panel: scene render, full height.
    ax_img = fig.add_axes([0.0, 0.0, IMG_W / VIDEO_W, 1.0])
    ax_img.set_axis_off()
    first = imageio.imread(os.path.join(FRAMES_DIR, "f00000.jpg"))
    im = ax_img.imshow(first, aspect="auto", interpolation="bilinear")
    hud = ax_img.text(0.03, 0.975, "", transform=ax_img.transAxes, fontsize=10,
                      color="white", va="top", ha="left", family="monospace",
                      bbox=dict(facecolor="black", alpha=0.45, pad=5,
                                edgecolor="none"))

    # Right top: running prediction error.
    ax_err = fig.add_axes([0.565, 0.60, 0.405, 0.33])
    ax_err.set_facecolor(C_SURFACE)
    for s in ("top", "right"):
        ax_err.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax_err.spines[s].set_color(C_MUTED)
    ax_err.tick_params(colors=C_TEXT2, labelsize=8.5)
    ax_err.set_xlim(0, times[-1])
    ymax = 1.1 * max(err_ema.max(), np.percentile(err_raw, 99.5))
    ax_err.set_ylim(0, ymax)
    ax_err.grid(True, axis="y", color=C_TRACK, lw=0.8)
    ax_err.set_axisbelow(True)
    ax_err.set_title("Prediction error (0.5 s ahead)  "
                     "$\\Vert y_k - \\hat{y}_{k|k-H} \\Vert$",
                     fontsize=11, color=C_TEXT, loc="left", pad=8)
    ax_err.set_xlabel("time [s]", fontsize=9, color=C_TEXT2)

    bump_band = ax_err.axvspan(0, 0, color=C_BAND, zorder=0, visible=False)
    bump_txt = ax_err.text(0, ymax * 0.96, "", fontsize=8.5, color=C_TEXT2,
                           ha="left", va="top", visible=False)
    (ln_raw,) = ax_err.plot([], [], color=C_BLUE, lw=0.8, alpha=0.30)
    (ln_ema,) = ax_err.plot([], [], color=C_BLUE, lw=2.0)
    ax_err.text(0.995, 1.02, "bold: moving average", transform=ax_err.transAxes,
                fontsize=8, color=C_TEXT2, ha="right", va="bottom")
    sw_line = ax_err.axvline(0, color=C_ORANGE, lw=1.4, ls=(0, (4, 3)),
                             visible=False)
    sw_txt = ax_err.text(0, ymax * 0.80, "", fontsize=8.5, color=C_ORANGE,
                         ha="left", va="top", visible=False)
    (cursor,) = ax_err.plot([], [], "o", ms=5, color=C_BLUE, zorder=5)

    # Right bottom: 8 CPG-parameter sliders.
    ax_sl = fig.add_axes([0.565, 0.05, 0.405, 0.46])
    ax_sl.set_axis_off()
    ax_sl.set_xlim(0, 1)
    ax_sl.set_ylim(-0.8, 8.4)
    ax_sl.invert_yaxis()
    ax_sl.text(0.0, -0.55, "CPG parameters", fontsize=11, color=C_TEXT,
               ha="left", va="center")
    mode_txt = ax_sl.text(1.0, -0.55, "", fontsize=9.5, ha="right", va="center",
                          color=C_TEXT2)

    TR_L, TR_R = 0.24, 0.86    # slider track extent in axes x
    knobs, val_txts = [], []
    for i, name in enumerate(PARAM_NAMES):
        y = i + 0.35
        ax_sl.text(TR_L - 0.02, y, name, fontsize=9, color=C_TEXT2,
                   ha="right", va="center")
        ax_sl.plot([TR_L, TR_R], [y, y], color=C_TRACK, lw=3.5,
                   solid_capstyle="round", zorder=1)
        # Faint destination tick: where the sloped optimum sits.
        xs = TR_L + norm(d["params_sloped"])[i] * (TR_R - TR_L)
        ax_sl.plot([xs, xs], [y - 0.16, y + 0.16], color=C_MUTED, lw=1.0,
                   zorder=2)
        (knob,) = ax_sl.plot([TR_L], [y], "o", ms=9, color=C_BLUE,
                             mec="white", mew=1.2, zorder=4)
        vt = ax_sl.text(TR_R + 0.025, y, "", fontsize=8.5, color=C_TEXT,
                        ha="left", va="center", family="monospace")
        knobs.append(knob)
        val_txts.append(vt)
    ax_sl.text(TR_L, 8.05, f"tick = sloped-terrain optimum", fontsize=8,
               color=C_MUTED, ha="left", va="center")

    fps = round(1.0 / (DT * RENDER_EVERY), 2)
    import subprocess
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(fps), "-i", "-",
         "-an", "-vcodec", "libopenh264", "-pix_fmt", "yuv420p", "-b:v", "5M",
         VIDEO_PATH],
        stdin=subprocess.PIPE)

    ramp_end = (switch_step + d["switch_ramp"]) if switch_step is not None else -1
    for fi, k in enumerate(frame_steps):
        im.set_data(imageio.imread(os.path.join(FRAMES_DIR, f"f{fi:05d}.jpg")))
        t = times[k]
        hud.set_text(f"t = {t:5.2f} s   forward = {d['pos_fwd'][k]:4.2f} m")

        ln_raw.set_data(times[:k + 1], err_raw[:k + 1])
        ln_ema.set_data(times[:k + 1], err_ema[:k + 1])
        cursor.set_data([t], [err_ema[k]])

        if t_enter is not None and t >= t_enter:
            x0, x1 = t_enter, min(t, t_exit)
            bump_band.set_visible(True)
            bump_band.set_x(x0)
            bump_band.set_width(max(x1 - x0, 1e-6))
            bump_txt.set_position((t_enter + 0.05, ymax * 0.96))
            bump_txt.set_text("on bump")
            bump_txt.set_visible(True)
        if t_switch is not None and t >= t_switch:
            sw_line.set_xdata([t_switch, t_switch])
            sw_line.set_visible(True)
            sw_txt.set_position((t_switch + 0.12, ymax * 0.82))
            sw_txt.set_text("switch to\nsloped params")
            sw_txt.set_visible(True)

        in_ramp = switch_step is not None and switch_step <= k < ramp_end
        knob_c = C_ORANGE if in_ramp else C_BLUE
        pn = norm(params[k])
        for i in range(8):
            knobs[i].set_xdata([TR_L + pn[i] * (TR_R - TR_L)])
            knobs[i].set_color(knob_c)
            knobs[i].set_markeredgecolor("white")
            val_txts[i].set_text(f"{params[k][i]:6.2f}")
        if d["mode"][k] == 1:
            mode_txt.set_text("parameter set: SLOPED-optimal")
            mode_txt.set_color(C_ORANGE)
        else:
            mode_txt.set_text("parameter set: FLAT-optimal")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:VIDEO_H, :VIDEO_W, :3]
        ff.stdin.write(np.ascontiguousarray(buf).tobytes())

    ff.stdin.close()
    ff.wait()
    plt.close(fig)
    print(f"[demo] wrote {VIDEO_PATH} ({len(frame_steps)} frames @ {fps} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--compose-only", action="store_true")
    ap.add_argument("--no-frames", action="store_true",
                    help="simulate without rendering (quick behavioral check)")
    args = ap.parse_args()

    if not args.compose_only:
        simulate(save_frames=not args.no_frames)
    if not args.sim_only and not args.no_frames:
        compose()
