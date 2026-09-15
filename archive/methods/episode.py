"""Shared CPG episode runner used by the terrain experiments.

One place defines how a bout is simulated (reset + jitter, CPG stepping, logging)
so the attitude convention and dynamics are identical everywhere. The previous
step's trunk roll/pitch is fed back to the CPG, so the VMC body-attitude feedback
is active whenever ``JointCPG.ATTITUDE_FEEDBACK`` is set (its default); with the
flag off the controller is open-loop. Attitude is in the PHYSICAL convention: with
the robot walking in +Y, getEulerFromQuaternion returns (forward nose-pitch,
lateral bank-roll, yaw), unpacked as pitch, roll, yaw.
"""

import numpy as np

DT = 0.01                 # control / simulation timestep [s]
SWITCH_RAMP_STEPS = 40    # parameter ramp length (0.4 s)
SLOPE_DEG = 10.0
N_COLS = 1600             # heightfield forward extent (+/- 40 m)
FAR_SLOPE_Y = 60.0        # slope start out of reach -> effectively flat
JITTER_STD = 0.002        # initial joint-angle jitter [rad]
TARGET_VX = 0.5           # v* [m/s]
ROBOT_MASS = 10.0
G = 9.81
DEFAULT_ORI = [0.0, 0.5, 0.5, 0.0]
LEG_NAMES = ["FL", "FR", "RL", "RR"]
PARAM_NAMES = ["coupling_gain", "w_swing", "w_stance", "F_FAST",
               "STOP_GAIN", "hip_amp", "knee_amp", "b"]


def _reset_with_jitter(p, robot, seed):
    """Reset to the stance pose, settle 1 s, with a seeded jitter on the initial
    joint angles (the only source of across-seed variability)."""
    from methods.marxefe_optimizer import JointCPG

    rng = np.random.default_rng(10_000 + int(seed))
    jit = rng.normal(0.0, JITTER_STD, size=12)

    p.resetBasePositionAndOrientation(robot, [0.0, 0.0, 0.55], DEFAULT_ORI)
    p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])
    abduction_ids, hip_ids, knee_ids = [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14]
    for i, jid in enumerate(abduction_ids):
        p.resetJointState(robot, jid, 0.0 + jit[i])
    for i, jid in enumerate(hip_ids):
        p.resetJointState(robot, jid, 0.05 + jit[4 + i])
    for i, jid in enumerate(knee_ids):
        p.resetJointState(robot, jid, -0.6 + jit[8 + i])

    for _ in range(int(1.0 / DT)):
        for jid in abduction_ids:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
        for jid in hip_ids:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, 0.25)
        for jid in knee_ids:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, -1.0)
        p.stepSimulation()
    return JointCPG(n_legs=4)


def run_episode(terrain_cfg, seed, params_start, params_target=None,
                switch_step=None, duration=20.0):
    """One episode on `terrain_cfg`. Holds `params_start`; if `params_target` is
    given, ramps to it over SWITCH_RAMP_STEPS starting at `switch_step`.
    Returns per-step logs (y forward, z height, x lateral, vx, roll, pitch, power)."""
    import pybullet as p
    from methods import terrain
    from methods.marxefe_optimizer import (check_if_fallen, get_base_orientation,
                                           load_environment, load_robot)

    terrain.TERRAIN_CONFIG = dict(terrain_cfg)
    load_environment(DT, use_gui=False)
    robot, _, joint_IDs_full, filtered, feet = load_robot(p)
    cpg = _reset_with_jitter(p, robot, seed)

    n_steps = int(round(duration / DT))
    log = {k: np.zeros(n_steps) for k in
           ["y", "z", "x", "vx", "roll", "pitch", "power"]}
    fell, fall_step = False, None
    actuated_ids = [1, 2, 5, 6, 9, 10, 13, 14]   # hip + knee per leg
    roll = pitch = 0.0                           # previous-step attitude for CPG feedback

    for k in range(n_steps):
        if params_target is not None and switch_step is not None and k >= switch_step:
            frac = min(1.0, (k - switch_step) / SWITCH_RAMP_STEPS)
            applied = params_start + frac * (params_target - params_start)
        else:
            applied = params_start

        raw = np.array([int(len(p.getContactPoints(
            bodyA=0, bodyB=robot, linkIndexA=-1, linkIndexB=feet[j])) > 0)
            for j in range(4)])
        hips, knees = cpg.step(applied, raw, DT, roll=roll, pitch=pitch)
        for j in range(4):
            abd, hip, knee = joint_IDs_full[LEG_NAMES[j]]
            p.setJointMotorControl2(robot, abd, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, hip, p.POSITION_CONTROL, hips[j])
            p.setJointMotorControl2(robot, knee, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()

        base_pos, base_ori = get_base_orientation(p, robot, DEFAULT_ORI)
        vel, _ = p.getBaseVelocity(robot)
        pitch, roll, _ = p.getEulerFromQuaternion(base_ori)  # physical convention (+Y forward)
        log["x"][k], log["y"][k], log["z"][k] = base_pos
        log["vx"][k] = vel[1]
        log["roll"][k], log["pitch"][k] = roll, pitch
        states = p.getJointStates(robot, actuated_ids)
        log["power"][k] = sum(abs(s[3] * s[1]) for s in states)

        is_fallen, _, _, _ = check_if_fallen(p, robot, base_pos, base_ori)
        if is_fallen:
            fell, fall_step = True, k
            for key in log:
                log[key] = log[key][:k + 1]
            break

    p.disconnect()
    log["fell"], log["fall_step"], log["n_steps"] = fell, fall_step, n_steps
    return log
