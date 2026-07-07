"""Capture PyBullet snapshots just before and after the friction terrain transition.

The robot walks on flat ground rendered as a checkerboard (standard friction).
A blue visual patch marks the low-friction ice zone.  Both effects use the
physics-correct friction terrain from methods/terrain.py; the blue box is
purely visual (no collision shape) and does not affect dynamics.

Screenshots taken via getCameraImage (software renderer, no display needed):
  * "before": robot approaching the ice patch on standard-friction ground
  * "after" : robot has crossed onto the ice patch

Output -> problem/figures/terrain_before.png, terrain_after.png

Run from the repository root:
    python problem/make_snapshots.py
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pybullet as p
import pybullet_data
from PIL import Image

from methods import terrain as terrain_mod
import methods.marxefe_optimizer as mxo

# ============================================================
# TERRAIN CONFIG
# ============================================================
ICE_START_Y  = 2.0          # y where ice patch starts [m]
ICE_LENGTH   = 4.0          # length of ice patch [m]
MU_NORMAL    = 0.7          # standard lateral friction
MU_ICE       = 0.30         # ice lateral friction (low enough to slip eventually)

TERRAIN_CFG = {
    "kind":   "friction",
    "zones":  [(ICE_START_Y, MU_ICE, "ice")],
    "base_mu": MU_NORMAL,
}

# ============================================================
# SIMULATION / SCREENSHOT CONFIG
# ============================================================
DT          = 0.01
MAX_STEPS   = 500

IMG_W, IMG_H = 960, 540

# Screenshot triggers relative to ice_start_y
BEFORE_OFFSET = -0.6        # robot at ICE_START_Y - 0.6 m  (on checkerboard)
AFTER_OFFSET  =  0.20       # robot at ICE_START_Y + 0.20 m (walking on ice)

# Camera — slightly elevated 3/4 view so checkerboard is visible
# Eye is behind and to the side of the robot; target is slightly ahead
CAM_DX =  3.0               # world-X offset of camera from robot centre
CAM_DY = -1.5               # world-Y offset (behind robot)
CAM_DZ =  2.0               # camera height
CAM_LOOK_DY = 0.5           # look-ahead of robot centre
CAM_LOOK_Z  = 0.0           # look-at height

# ============================================================
# CPG CONSTANTS  (match marxefe_optimizer.py exactly)
# ============================================================
CPG_PARAMS = np.array([9.09, 14.05, 10.61, 25.58, 0.42, 0.33, 0.80, 7.32])

ALPHA, BETA  = 3.0, 12.0
U_SQ         = 2.0
HIP_OFF      = 0.26
KNEE_OFF     = -1.0
N_LEGS       = 4
K_MATRIX     = np.array([
    [ 0, -1, -1,  1],
    [-1,  0,  1, -1],
    [-1,  1,  0, -1],
    [ 1, -1, -1,  0],
], dtype=float)

LEG_NAMES      = ["FL", "FR", "RL", "RR"]
JOINT_IDS_FULL = {"FL": [4, 5, 6], "FR": [0, 1, 2],
                  "RL": [12, 13, 14], "RR": [8, 9, 10]}
FEET_IDS       = [7, 3, 15, 11]
ABD_IDS        = [0, 4, 8, 12]
HIP_IDS        = [1, 5, 9, 13]
KNEE_IDS       = [2, 6, 10, 14]
DEFAULT_ORI    = [0.0, 0.5, 0.5, 0.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ============================================================
# Helpers
# ============================================================
def add_ice_visual(ice_start, ice_len, half_width=8.0):
    """Create a thin, collision-free blue box marking the ice zone."""
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half_width, ice_len / 2.0, 0.002],
        rgbaColor=[0.15, 0.45, 0.95, 0.85],
    )
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,   # no collision
        baseVisualShapeIndex=vis,
        basePosition=[0.0, ice_start + ice_len / 2.0, 0.001],
        baseOrientation=[0, 0, 0, 1],
    )
    return body


def render_frame(robot_y, label):
    """Capture one screenshot with a 3/4 overhead view."""
    eye    = [CAM_DX,  robot_y + CAM_DY,  CAM_DZ]
    target = [0.0,     robot_y + CAM_LOOK_DY,  CAM_LOOK_Z]
    vm = p.computeViewMatrix(eye, target, [0, 0, 1])
    pm = p.computeProjectionMatrixFOV(
        fov=50, aspect=IMG_W / IMG_H, nearVal=0.1, farVal=30.0
    )
    _, _, rgba, _, _ = p.getCameraImage(
        IMG_W, IMG_H, viewMatrix=vm, projectionMatrix=pm,
        renderer=p.ER_TINY_RENDERER,
    )
    arr  = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)
    img  = Image.fromarray(arr)
    path = os.path.join(OUT_DIR, f"terrain_{label}.png")
    img.save(path)
    print(f"  [{label}] robot_y={robot_y:.2f} m  ->  {path}")
    return path


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    terrain_mod.TERRAIN_CONFIG = TERRAIN_CFG

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)

    # Ground: plane.urdf gives checkerboard texture; friction managed per-step
    gid = terrain_mod.build_ground(p)   # body 0, flat plane.urdf

    # Visual ice patch (blue box, no collision)
    add_ice_visual(ICE_START_Y, ICE_LENGTH)

    # Robot
    robot = p.loadURDF(
        "laikago/laikago_toes.urdf",
        [0.0, 0.0, 0.55], DEFAULT_ORI,
        flags=p.URDF_USE_SELF_COLLISION,
        useFixedBase=False,
    )

    # ---- Settle (100 steps) ----------------------------------------
    for jid in ABD_IDS:
        p.resetJointState(robot, jid, 0.0)
    for jid in HIP_IDS:
        p.resetJointState(robot, jid, 0.05)
    for jid in KNEE_IDS:
        p.resetJointState(robot, jid, -0.6)
    for _ in range(100):
        for jid in ABD_IDS:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
        for jid in HIP_IDS:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, 0.25)
        for jid in KNEE_IDS:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, -1.0)
        p.stepSimulation()

    # ---- CPG init (on limit cycle, trot phases) --------------------
    r0    = np.sqrt(U_SQ)
    theta = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    cx    = np.array([r0 * np.cos(t) for t in theta])
    cy    = np.array([r0 * np.sin(t) for t in theta])

    mxo.phase_state_memory = []
    for j in range(N_LEGS):
        if   cy[j] > mxo.SWING_ENTER:  mxo.phase_state_memory.append("swing")
        elif cy[j] < mxo.STANCE_ENTER: mxo.phase_state_memory.append("stance")
        else:                           mxo.phase_state_memory.append("transition")

    cg, ws, wst, FF, sg, ha, ka, b = CPG_PARAMS

    debounced  = np.zeros(N_LEGS, dtype=int)
    change_cnt = np.zeros(N_LEGS, dtype=int)

    saved = {}
    print(f"\nRunning (ice at y={ICE_START_Y:.1f} m, "
          f"mu_normal={MU_NORMAL}, mu_ice={MU_ICE}) ...")

    for k in range(MAX_STEPS):
        base_pos, _ = p.getBasePositionAndOrientation(robot)
        robot_y = base_pos[1]

        if k % 100 == 0:
            print(f"  step {k:3d}  y={robot_y:.3f}  z={base_pos[2]:.3f}")

        # Screenshot triggers
        if "before" not in saved and robot_y >= ICE_START_Y + BEFORE_OFFSET:
            saved["before"] = render_frame(robot_y, "before")
        if "after" not in saved and robot_y >= ICE_START_Y + AFTER_OFFSET:
            saved["after"] = render_frame(robot_y, "after")
        if len(saved) == 2:
            print("  Both shots captured.")
            break

        # ---- CPG step (loop-based, matches marxefe_optimizer.py) ---
        w_vec = np.zeros(N_LEGS)
        r_vec = np.zeros(N_LEGS)
        for j in range(N_LEGS):
            w = wst / (np.exp(-b * cy[j]) + 1) + ws / (np.exp(b * cy[j]) + 1)
            w_vec[j] = w
            r = np.sqrt(cx[j]**2 + cy[j]**2)
            r_vec[j] = r
            cx[j] += DT * (ALPHA * (U_SQ - r**2) * cx[j] - w * cy[j])

        raw = np.array([
            int(len(p.getContactPoints(
                bodyA=gid, bodyB=robot, linkIndexA=-1, linkIndexB=fid)) > 0)
            for fid in FEET_IDS
        ])
        for j in range(N_LEGS):
            if raw[j] == debounced[j]:
                change_cnt[j] = 0
            else:
                change_cnt[j] += 1
                if change_cnt[j] >= 2:
                    debounced[j] = raw[j]
                    change_cnt[j] = 0
        phases = [mxo.get_phase(cy[j], j) for j in range(N_LEGS)]
        u_fb, _ = mxo.compute_feedback_u(
            cx, cy, w_vec, K_MATRIX, debounced, phases,
            FF, 0.5, 0.5, cg, sg,
        )
        for j in range(N_LEGS):
            coupling = cg * np.dot(K_MATRIX[j, :], cy)
            cy[j] += DT * (
                BETA * (U_SQ - r_vec[j]**2) * cy[j]
                + w_vec[j] * cx[j] + coupling + u_fb[j]
            )

        for j, leg in enumerate(LEG_NAMES):
            abd, hip, kn = JOINT_IDS_FULL[leg]
            p.setJointMotorControl2(robot, abd, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, hip, p.POSITION_CONTROL,
                                    HIP_OFF + ha * cx[j])
            p.setJointMotorControl2(robot, kn,  p.POSITION_CONTROL,
                                    KNEE_OFF - ka * max(0.0, cy[j]))

        terrain_mod.apply_dynamic_friction(p, robot, robot_y)
        p.stepSimulation()

    p.disconnect()

    if len(saved) < 2:
        missing = [s for s in ("before", "after") if s not in saved]
        print(f"\nWarning: missing shots: {missing}  (final y={robot_y:.2f})")
    else:
        print(f"\nDone. Written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
