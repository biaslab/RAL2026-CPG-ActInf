"""Bayesian optimization with a Gaussian-process surrogate and UCB acquisition.

Provides:
  * BetaSchedule, BOOptimizer — the generic GP-UCB optimizer (any black-box
    `evaluate` callable mapping a parameter vector → scalar J to maximize).
  * CPG-specific pipeline (run_cpg_trial, compute_objective, evaluate_candidate,
    bo_optimize_cpg) for tuning Righetti-style central pattern generator
    parameters on the Laikago quadruped in PyBullet.

The CPG pipeline shares its 4-D observation [pos_x, pos_y, pitch, roll] and its
position-based goal (target_forward_position, 0, 0, 0) with the MARXEFE active
inference baseline so the two can be compared directly.
"""

import csv
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from methods.cpg_bounds import bounds, bounds_lower, bounds_upper


# ============================================================================
# GENERIC BAYESIAN OPTIMIZER
# ============================================================================

@dataclass
class BetaSchedule:
    """Decaying exploration coefficient for the UCB acquisition function."""

    beta_init: float = 5.0
    beta_min: float = 0.5
    n_decay_start: int = 10
    gamma: float = 0.7

    def __call__(self, t: int) -> float:
        if t <= self.n_decay_start:
            return self.beta_init
        decayed = self.beta_init * (self.gamma ** (t - self.n_decay_start))
        return max(self.beta_min, decayed)


class BOOptimizer:
    """Online Bayesian optimization of a black-box objective.

    Parameters
    ----------
    bounds : torch.Tensor
        Tensor of shape (2, d) with lower and upper parameter bounds.
    beta_schedule : BetaSchedule
        Schedule controlling exploration vs. exploitation in UCB.
    n_init : int
        Number of uniformly random initial trials before fitting the GP.
    seed : int
        Random seed for reproducibility of the random initialization.

    Notes
    -----
    The GP is fit on parameters normalized to the unit cube and on standardized
    objective values. Candidate selection uses UpperConfidenceBound with a
    decaying beta. The class is independent of the simulation back-end: pass any
    `evaluate` callable to :meth:`optimize`.
    """

    def __init__(
        self,
        bounds: torch.Tensor,
        beta_schedule: Optional[BetaSchedule] = None,
        n_init: int = 5,
        seed: int = 0,
    ):
        self.bounds = bounds.to(dtype=torch.double)
        self.lower = self.bounds[0]
        self.upper = self.bounds[1]
        self.d = self.bounds.shape[1]
        self.unit_bounds = torch.tensor(
            [[0.0] * self.d, [1.0] * self.d], dtype=torch.double
        )
        self.beta_schedule = beta_schedule or BetaSchedule()
        self.n_init = n_init
        self.seed = seed

        self.train_X_unit = torch.empty(0, self.d, dtype=torch.double)
        self.train_X = torch.empty(0, self.d, dtype=torch.double)
        self.train_Y = torch.empty(0, 1, dtype=torch.double)

    # ---------- normalization ----------

    def to_unit(self, x_np: np.ndarray) -> torch.Tensor:
        x = torch.tensor(x_np, dtype=torch.double)
        return (x - self.lower) / (self.upper - self.lower)

    def from_unit(self, x_unit: torch.Tensor) -> np.ndarray:
        return (x_unit * (self.upper - self.lower) + self.lower).detach().numpy()

    # ---------- model + acquisition ----------

    def fit_model(self) -> SingleTaskGP:
        Y = self.train_Y
        Y_mean = Y.mean()
        Y_std = Y.std().clamp_min(1e-6)
        model = SingleTaskGP(self.train_X_unit, (Y - Y_mean) / Y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.Y_mean_ = Y_mean
        model.Y_std_ = Y_std
        return model

    def suggest(self, model: SingleTaskGP, beta_t: float) -> torch.Tensor:
        acqf = UpperConfidenceBound(model, beta=beta_t)
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.unit_bounds,
            q=1,
            num_restarts=10,
            raw_samples=64,
        )
        return candidate.squeeze(0)

    # ---------- main loop ----------

    def optimize(
        self,
        evaluate: Callable[[np.ndarray, int], float],
        n_trials: int,
        on_trial: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ):
        """Run BO for `n_trials` evaluations.

        Parameters
        ----------
        evaluate : callable (params, trial_idx) -> float
            Black-box objective; should return the scalar J to maximize.
        n_trials : int
            Total number of trials, including the random init phase.
        on_trial : optional callable (trial_idx, params, J) -> None
            Hook called after each trial; useful for CSV logging.
        """
        rng = np.random.default_rng(self.seed)

        # phase 1: random initialization
        for i in range(self.n_init):
            x_np = rng.uniform(self.lower.numpy(), self.upper.numpy())
            J = evaluate(x_np, i + 1)
            self._append(x_np, J)
            if on_trial is not None:
                on_trial(i + 1, x_np, J)

        # phase 2: GP-UCB loop
        for t in range(self.n_init, n_trials):
            model = self.fit_model()
            beta_t = self.beta_schedule(t)
            x_unit = self.suggest(model, beta_t)
            x_np = self.from_unit(x_unit)
            J = evaluate(x_np, t + 1)
            self._append(x_np, J)
            if on_trial is not None:
                on_trial(t + 1, x_np, J)

        return self.best()

    def _append(self, x_np: np.ndarray, J: float) -> None:
        x_torch = torch.tensor(x_np, dtype=torch.double).unsqueeze(0)
        self.train_X = torch.cat([self.train_X, x_torch], dim=0)
        self.train_X_unit = torch.cat([self.train_X_unit, self.to_unit(x_np).unsqueeze(0)], dim=0)
        self.train_Y = torch.cat(
            [self.train_Y, torch.tensor([[J]], dtype=torch.double)], dim=0
        )

    def best(self):
        idx = int(self.train_Y.argmax().item())
        return self.train_X[idx].numpy(), float(self.train_Y[idx].item())


# ============================================================================
# GUI / DIRECT FLAG
# Set USE_GUI = False to run headless (DIRECT mode, faster, no display needed).
# Set USE_GUI = True  to open the PyBullet GUI window.
# ============================================================================

USE_GUI = False

# ============================================================================
# PYBULLET SETUP FUNCTIONS
# ============================================================================

quadruped = None


def load_environment(dt, use_gui=USE_GUI):
    """Initialize PyBullet environment."""
    if use_gui:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, True)
        p.resetDebugVisualizerCamera(
            cameraDistance=5, cameraYaw=90,
            cameraPitch=-20, cameraTargetPosition=[0, 0, 0.6]
        )
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(dt)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf")
    return p


def load_robot(p):
    """Load Laikago robot and configure collision filtering."""
    start_position = [0.0, 0.0, 0.55]
    start_orientation = [0.0, 0.5, 0.5, 0.0]
    urdfFlags = p.URDF_USE_SELF_COLLISION
    use_rack = False

    quadruped = p.loadURDF("laikago/laikago_toes.urdf", start_position, start_orientation, flags=urdfFlags, useFixedBase=use_rack)
    p.changeDynamics(quadruped, -1, mass=10.0)

    n_joints = p.getNumJoints(quadruped)
    joints_info = {}
    print("\n=== Laikago Joint Info ===")
    lower_legs = []

    for i in range(n_joints):
        joint_info = p.getJointInfo(quadruped, i)
        joint_ID, joint_name, joint_type, joint_qID, joint_uID, joint_flags, joint_damping, joint_friction, joint_lower_limit, joint_upper_limit, joint_max_force, joint_max_velocity, joint_link_name, joint_axis, joint_parent_frame_pos, joint_parent_frame_ori, joint_parent_ID = joint_info
        joint_name = joint_name.decode('utf-8')
        joint_link_name = joint_link_name.decode('utf-8')
        joints_info[joint_ID] = dict(
            joint_name=joint_name,
            joint_type=joint_type,
            joint_qID=joint_qID,
            joint_uID=joint_uID,
            joint_flags=joint_flags,
            joint_damping=joint_damping,
            joint_friction=joint_friction,
            joint_lower_limit=joint_lower_limit,
            joint_upper_limit=joint_upper_limit,
            joint_max_force=joint_max_force,
            joint_max_velocity=joint_max_velocity,
            joint_link_name=joint_link_name,
            joint_axis=joint_axis,
            joint_parent_frame_pos=joint_parent_frame_pos,
            joint_parent_frame_ori=joint_parent_frame_ori,
            joint_parent_ID=joint_parent_ID
        )

        if 'lower_leg' in joint_link_name:
            lower_legs.append(joint_ID)

    for joint_ID, joint_info in joints_info.items():
        joint_link_name = joint_info['joint_link_name']
        joint_parent_ID = joint_info['joint_parent_ID']
        joint_parent_link_name = 'chassis' if joint_parent_ID == -1 else joints_info[joint_parent_ID]['joint_link_name']
        print(f'{joint_ID} {joint_link_name} -> {joint_parent_link_name}')

    for l0 in lower_legs:
        for l1 in lower_legs:
            if l0 == l1:
                continue
            p.setCollisionFilterPair(quadruped, quadruped, l0, l1, 1)

    n_legs = 4
    feet_joint_IDs = [7, 3, 15, 11]
    joint_IDs_full = {
        'FL': [4, 5, 6],
        'FR': [0, 1, 2],
        'RL': [12, 13, 14],
        'RR': [8, 9, 10]
    }

    return quadruped, n_legs, joints_info, joint_IDs_full, feet_joint_IDs


# ============================================================================
# TRIAL WRAPPER
# ============================================================================

_prev_params = None


def run_cpg_trial(params: np.ndarray,
                  target_forward_position: float,
                  context_mode: str = "flat") -> dict:
    """Run a single CPG-controlled locomotion trial with MARXEFE-style reset."""
    global _prev_params, quadruped

    if quadruped is None:
        dt = 0.01
        load_environment(dt)
        quadruped, _, _, _, _ = load_robot(p)
        print(f"[run_cpg_trial] Initialized environment with robot ID: {quadruped}")

    # Unpack parameters
    coupling_gain = params[0]
    w_swing = params[1]
    w_stance = params[2]
    F_FAST = params[3]
    STOP_GAIN = params[4]
    hip_amplitude = params[5]
    knee_amplitude = params[6]
    b = params[7]

    alpha = 3.0
    beta = 12.0
    u = 2.0
    hip_offset = 0.26
    knee_offset = -1.0

    dt = 0.01
    trial_duration = 4.5
    trial_steps = int(trial_duration / dt)
    transition_duration = 1.5
    transition_steps = int(transition_duration / dt)

    DEFAULT_ORI = [0.0, 0.5, 0.5, 0.0]

    # ========================================================================
    # RESET TO MARXEFE-STYLE POSE
    # ========================================================================

    start_position = [0.0, 0.0, 0.55]
    p.resetBasePositionAndOrientation(quadruped, start_position, DEFAULT_ORI)
    p.resetBaseVelocity(quadruped, [0, 0, 0], [0, 0, 0])

    abduction_joint_ids = [0, 4, 8, 12]
    hip_joint_ids = [1, 5, 9, 13]
    knee_joint_ids = [2, 6, 10, 14]

    # Reset to neutral pose matching MARXEFE
    for jid in abduction_joint_ids:
        p.resetJointState(quadruped, jid, 0.0)
    for jid in hip_joint_ids:
        p.resetJointState(quadruped, jid, 0.05)
    for jid in knee_joint_ids:
        p.resetJointState(quadruped, jid, -0.6)

    # PRE-TRIAL SETTLING (OUTSIDE 4.5s trial)
    for settle_step in range(100):
        for jid in abduction_joint_ids:
            p.setJointMotorControl2(
                quadruped, jid, p.POSITION_CONTROL,
                targetPosition=0.0, force=500
            )
        for jid in hip_joint_ids:
            p.setJointMotorControl2(quadruped, jid, p.POSITION_CONTROL, 0.25)
        for jid in knee_joint_ids:
            p.setJointMotorControl2(quadruped, jid, p.POSITION_CONTROL, -1.0)
        p.stepSimulation()

    # Initialize CPG on limit cycle
    n_legs = 4
    cpg_x = np.zeros(n_legs)
    cpg_y = np.zeros(n_legs)
    theta = [0, np.pi/2, np.pi, 3*np.pi/2]
    for i in range(n_legs):
        cpg_x[i] = np.sqrt(u) * np.cos(theta[i])
        cpg_y[i] = np.sqrt(u) * np.sin(theta[i])

    k = np.array([
        [0, -1, -1, 1],
        [-1, 0, 1, -1],
        [-1, 1, 0, -1],
        [1, -1, -1, 0]
    ])

    # Phase thresholds matching MARXEFE
    SWING_ENTER = 0.15
    SWING_EXIT = 0.02
    STANCE_ENTER = -0.15
    STANCE_EXIT = -0.02

    phase_state_memory = []
    for j in range(n_legs):
        if cpg_y[j] > SWING_ENTER:
            phase_state_memory.append("swing")
        elif cpg_y[j] < STANCE_ENTER:
            phase_state_memory.append("stance")
        else:
            phase_state_memory.append("transition")

    feet_joint_IDs = [7, 3, 15, 11]
    joint_IDs_full = {
        "FL": [4, 5, 6],
        "FR": [0, 1, 2],
        "RL": [12, 13, 14],
        "RR": [8, 9, 10]
    }

    leg_names = ["FL", "FR", "RL", "RR"]

    # Logging arrays (trial only)
    time_log = np.zeros(trial_steps)
    x_log = np.zeros((n_legs, trial_steps))
    y_log = np.zeros((n_legs, trial_steps))
    vx_log = np.zeros(trial_steps)
    vy_log = np.zeros(trial_steps)
    roll_log = np.zeros(trial_steps)
    pitch_log = np.zeros(trial_steps)
    yaw_log = np.zeros(trial_steps)
    contact_forces_log = np.zeros((n_legs, trial_steps))
    n_joints = 2 * n_legs
    torques_log = np.zeros((n_joints, trial_steps))
    qdot_log = np.zeros((n_joints, trial_steps))
    base_pos_log = np.zeros((trial_steps, 3))

    fall_detected = False
    debounced_contacts = np.zeros(n_legs, dtype=int)
    contact_change_count = np.zeros(n_legs, dtype=int)
    DEBOUNCE_THRESHOLD = 2

    # Parameter trajectory
    if _prev_params is None:
        params_trajectory = np.tile(params, (trial_steps, 1))
    else:
        interp_params = np.zeros((transition_steps, len(params)))
        for i in range(len(params)):
            interp_params[:, i] = np.linspace(_prev_params[i], params[i], transition_steps)
        steady_params = np.tile(params, (trial_steps - transition_steps, 1))
        params_trajectory = np.vstack([interp_params, steady_params])
    _prev_params = params.copy()

    # Helper functions
    def get_phase(y_val, leg_idx):
        current_state = phase_state_memory[leg_idx]
        if current_state == "swing":
            new_state = "transition" if y_val < SWING_EXIT else "swing"
        elif current_state == "stance":
            new_state = "transition" if y_val > STANCE_EXIT else "stance"
        else:
            if y_val > SWING_ENTER:
                new_state = "swing"
            elif y_val < STANCE_ENTER:
                new_state = "stance"
            else:
                new_state = "transition"
        phase_state_memory[leg_idx] = new_state
        return new_state

    def compute_feedback_u(x_vec, y_vec, w_vec, contacts, phases,
                           F_fast, STOP_gain, coupling_gain_val):
        n = len(x_vec)
        u_fb = np.zeros(n)
        coupling_y = coupling_gain_val * (k @ y_vec)
        for j in range(n):
            contact = contacts[j]
            phase = phases[j]
            if phase == "swing":
                if contact < 0.5:
                    u_fb[j] = STOP_gain * (w_vec[j] * x_vec[j] - coupling_y[j])
                else:
                    u_fb[j] = np.sign(y_vec[j]) * F_fast
            elif phase == "stance":
                if contact > 0.5:
                    u_fb[j] = STOP_gain * (w_vec[j] * x_vec[j] - coupling_y[j])
                else:
                    u_fb[j] = np.sign(y_vec[j]) * F_fast
            else:
                u_fb[j] = 0.0
        return u_fb

    def check_if_fallen(base_pos, base_orientation):
        rot_mat = p.getMatrixFromQuaternion(base_orientation)
        local_up = np.array(rot_mat[6:])
        world_up = np.array([0, 0, 1])
        magic_value = np.dot(world_up, local_up)
        fallen_ori = magic_value < 0.3
        fallen_height = base_pos[2] < 0.25
        return fallen_ori or fallen_height

    # ========================================================================
    # TRIAL SIMULATION
    # ========================================================================

    x_log[:, 0] = cpg_x
    y_log[:, 0] = cpg_y

    for step in range(1, trial_steps):
        current_time = step * dt
        time_log[step] = current_time

        current_params = params_trajectory[step]
        coupling_gain_curr = current_params[0]
        w_swing_curr = current_params[1]
        w_stance_curr = current_params[2]
        F_FAST_curr = current_params[3]
        STOP_GAIN_curr = current_params[4]
        hip_amplitude_curr = current_params[5]
        knee_amplitude_curr = current_params[6]
        b_curr = current_params[7]

        # CPG dynamics
        w_vec = np.zeros(n_legs)
        r_vec = np.zeros(n_legs)
        for j in range(n_legs):
            y_prev = y_log[j, step-1]
            x_prev = x_log[j, step-1]
            w = (w_stance_curr / (np.exp(-b_curr * y_prev) + 1) +
                 w_swing_curr / (np.exp(b_curr * y_prev) + 1))
            w_vec[j] = w
            r = np.sqrt(x_prev**2 + y_prev**2)
            r_vec[j] = r
            dx = alpha * (u - r**2) * x_prev - w * y_prev
            x_log[j, step] = x_prev + dt * dx

        # Contact sensing
        raw_contacts = np.array([
            int(len(p.getContactPoints(bodyA=0, bodyB=quadruped,
                                       linkIndexA=-1, linkIndexB=ID)) > 0)
            for ID in feet_joint_IDs
        ])
        for j in range(n_legs):
            if raw_contacts[j] == debounced_contacts[j]:
                contact_change_count[j] = 0
            else:
                contact_change_count[j] += 1
            if contact_change_count[j] >= DEBOUNCE_THRESHOLD:
                debounced_contacts[j] = raw_contacts[j]
                contact_change_count[j] = 0

        phases = [get_phase(y_log[j, step-1], j) for j in range(n_legs)]
        u_fb = compute_feedback_u(
            x_log[:, step-1], y_log[:, step-1], w_vec,
            debounced_contacts, phases,
            F_FAST_curr, STOP_GAIN_curr, coupling_gain_curr
        )

        for j in range(n_legs):
            y_prev = y_log[j, step-1]
            x_prev = x_log[j, step-1]
            r = r_vec[j]
            w = w_vec[j]
            coupling_term = coupling_gain_curr * np.dot(k[j, :], y_log[:, step-1])
            dy = beta * (u - r**2) * y_prev + w * x_prev + coupling_term + u_fb[j]
            y_log[j, step] = y_prev + dt * dy

        # Joint control
        for j in range(n_legs):
            leg_name = leg_names[j]
            abd_joint, hip_joint, knee_joint = joint_IDs_full[leg_name]
            p.setJointMotorControl2(quadruped, abd_joint, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            hip_angle = hip_offset + hip_amplitude_curr * x_log[j, step]
            knee_angle = knee_offset - knee_amplitude_curr * max(0, y_log[j, step])
            p.setJointMotorControl2(quadruped, hip_joint, p.POSITION_CONTROL, hip_angle)
            p.setJointMotorControl2(quadruped, knee_joint, p.POSITION_CONTROL, knee_angle)

        p.stepSimulation()

        # Logging
        joint_idx = 0
        for jid_hip, jid_knee in zip(hip_joint_ids, knee_joint_ids):
            hip_state = p.getJointState(quadruped, jid_hip)
            knee_state = p.getJointState(quadruped, jid_knee)
            torques_log[joint_idx, step] = hip_state[3]
            torques_log[joint_idx+1, step] = knee_state[3]
            qdot_log[joint_idx, step] = hip_state[1]
            qdot_log[joint_idx+1, step] = knee_state[1]
            joint_idx += 2

        base_vel, _ = p.getBaseVelocity(quadruped)
        vx_log[step] = base_vel[1]   # forward (+Y) - matches MARXEFE
        vy_log[step] = base_vel[0]   # lateral  (X) - matches MARXEFE

        base_pos, base_quat = p.getBasePositionAndOrientation(quadruped)
        base_pos_log[step, :] = base_pos

        _, base_orientation = p.multiplyTransforms(
            [0, 0, 0], base_quat, [0, 0, 0], DEFAULT_ORI
        )
        euler = p.getEulerFromQuaternion(base_orientation)
        roll_log[step] = euler[0]
        pitch_log[step] = euler[1]
        yaw_log[step] = euler[2]

        for j in range(n_legs):
            contact_points = p.getContactPoints(
                bodyA=0, bodyB=quadruped,
                linkIndexB=feet_joint_IDs[j]
            )
            if len(contact_points) > 0:
                contact_forces_log[j, step] = sum([cp[9] for cp in contact_points])

        if check_if_fallen(base_pos, base_orientation):
            fall_detected = True
            actual_steps = step + 1
            time_log = time_log[:actual_steps]
            x_log = x_log[:, :actual_steps]
            y_log = y_log[:, :actual_steps]
            vx_log = vx_log[:actual_steps]
            vy_log = vy_log[:actual_steps]
            roll_log = roll_log[:actual_steps]
            pitch_log = pitch_log[:actual_steps]
            yaw_log = yaw_log[:actual_steps]
            contact_forces_log = contact_forces_log[:, :actual_steps]
            torques_log = torques_log[:, :actual_steps]
            qdot_log = qdot_log[:, :actual_steps]
            base_pos_log = base_pos_log[:actual_steps, :]
            break

    # ========================================================================
    # POST-PROCESSING: COMPUTE METRICS
    # ========================================================================

    start_idx = int(transition_duration / dt)

    # Corrected forward distance and lateral drift assuming forward motion along +Y
    if len(base_pos_log) > 0:
        forward_distance = base_pos_log[-1, 1] - base_pos_log[0, 1]   # Y-axis forward
        lateral_drift = abs(base_pos_log[-1, 0] - base_pos_log[0, 0])  # X-axis sideways
    else:
        forward_distance = 0.0
        lateral_drift = 0.0

    # Stability computed on steady-state
    if len(roll_log) > start_idx:
        roll_window = roll_log[start_idx:]
        pitch_window = pitch_log[start_idx:]
        rms_roll_deg = np.rad2deg(np.sqrt(np.mean(roll_window**2)))
        rms_pitch_deg = np.rad2deg(np.sqrt(np.mean(pitch_window**2)))
        combined_stability = np.sqrt(rms_roll_deg**2 + rms_pitch_deg**2)
    else:
        combined_stability = 1000.0

    # Mean forward velocity along +Y (forward axis), computed over steady-state
    if len(base_pos_log) > start_idx:
        y0 = base_pos_log[start_idx, 1]
        yT = base_pos_log[-1, 1]
        T_steady = (len(base_pos_log) - start_idx) * dt
        mean_vx = (yT - y0) / T_steady if T_steady > 0 else 0.0
    else:
        mean_vx = 0.0

    return {
        "t": time_log,
        "pos_x": base_pos_log[:, 1],   # forward +Y
        "pos_y": base_pos_log[:, 0],   # lateral  X
        "vx": vx_log,
        "vy": vy_log,
        "roll": roll_log,
        "pitch": pitch_log,
        "yaw": yaw_log,
        "forces": contact_forces_log.T,
        "torques": torques_log.T,
        "qdot": qdot_log.T,
        "base_pos": base_pos_log,
        "fall": fall_detected,
        "transition_duration": transition_duration,
        "x_cpg": x_log,
        "y_cpg": y_log,
        "forward_distance": forward_distance,
        "lateral_drift": lateral_drift,
        "stability": combined_stability,
        "mean_vx": mean_vx
    }


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def compute_objective(trial_data: dict,
                      target_forward_position: float,
                      robot_mass: float,
                      g: float = 9.81) -> float:
    """Compute J = position_reward - w2*CoT_norm - w3*stability.

    Position reward rewards closeness of (pos_x, pos_y) to
    (target_forward_position, 0) across the steady-state window. Mirrors the
    MARXEFE goal prior so BO and MARXEFE optimise the same signal.
    """
    t = trial_data["t"]
    transition_duration = trial_data["transition_duration"]
    steady_idx = np.searchsorted(t, transition_duration)

    if steady_idx >= len(t) - 5:
        return -50.0

    t_steady = t[steady_idx:]
    pos_x_steady = trial_data["pos_x"][steady_idx:]
    pos_y_steady = trial_data["pos_y"][steady_idx:]
    torques_steady = trial_data["torques"][steady_idx:, :]
    qdot_steady = trial_data["qdot"][steady_idx:, :]
    base_pos_steady = trial_data["base_pos"][steady_idx:, :]

    T = len(t_steady)

    # Position reward (matches MARXEFE goal prior over [pos_x, pos_y])
    w1 = 5.0
    l_r = 0.85
    sigma_pos_x = 1.0   # forward position tolerance (m)
    sigma_pos_y = 0.1   # lateral  position tolerance (m)
    R_p_sum = 0.0
    for i in range(T):
        err_sq = ((pos_x_steady[i] - target_forward_position) / sigma_pos_x)**2 \
               + (pos_y_steady[i] / sigma_pos_y)**2
        reward_i = np.exp(-0.5 * err_sq)
        R_p_sum += min(reward_i, l_r)
    R_p = R_p_sum / T
    position_term = w1 * R_p

    # Cost of Transport
    w2 = 0.4
    delta_t = 0.02
    mechanical_power = np.sum(np.abs(torques_steady * qdot_steady)) * delta_t
    d = base_pos_steady[-1, 1] - base_pos_steady[0, 1]
    if d < 0.5:
        CoT_cap = 200.0
    elif d < 1.5:
        CoT_cap = 150.0
    else:
        CoT_cap = 100.0

    if d < 0.005:
        CoT = mechanical_power / (robot_mass * g * max(d, 0.001))
    else:
        CoT = mechanical_power / (robot_mass * g * d)

    CoT = min(CoT, CoT_cap)
    CoT_norm = CoT / 50.0

    stability = trial_data["stability"]
    w3 = 0.02

    J = position_term - w2 * CoT_norm - w3 * stability
    return J


# ============================================================================
# EVALUATION FUNCTION (metrics keys aligned to MARXEFE CSV schema)
# ============================================================================

def evaluate_candidate(params_np, target_forward_position, robot_mass, optimizer_name, seed, trial_idx):
    """Evaluate candidate parameters and return standardized metrics."""
    sim_start = time.time()

    trial_data = run_cpg_trial(params_np, target_forward_position, context_mode="flat")
    sim_time_sec = time.time() - sim_start

    J = compute_objective(trial_data, target_forward_position, robot_mass)

    forward_distance = trial_data["forward_distance"]
    lateral_drift    = trial_data["lateral_drift"]
    fell             = int(trial_data["fall"])
    stability        = trial_data["stability"]
    mean_vx          = trial_data["mean_vx"]

    # CoT (logged separately, with same caps as MARXEFE)
    t               = trial_data["t"]
    transition_duration = trial_data["transition_duration"]
    steady_idx      = np.searchsorted(t, transition_duration)

    if steady_idx < len(t) - 5:
        torques_steady    = trial_data["torques"][steady_idx:, :]
        qdot_steady       = trial_data["qdot"][steady_idx:, :]
        base_pos_steady   = trial_data["base_pos"][steady_idx:, :]
        dt_val            = t[1] - t[0] if len(t) > 1 else 0.01
        mechanical_power  = np.sum(np.abs(torques_steady * qdot_steady)) * dt_val
        fwd_dist          = base_pos_steady[-1, 1] - base_pos_steady[0, 1]
        if fwd_dist < 0.5:
            CoT_cap = 200.0
        elif fwd_dist < 1.5:
            CoT_cap = 150.0
        else:
            CoT_cap = 100.0
        if fwd_dist > 0.001:
            CoT = mechanical_power / (robot_mass * 9.81 * fwd_dist)
        else:
            CoT = CoT_cap
        CoT = min(CoT, CoT_cap)
    else:
        CoT = 1000.0

    # RMS roll / pitch (steady-state window, degrees)
    if len(trial_data["roll"]) > steady_idx:
        roll_window  = trial_data["roll"][steady_idx:]
        pitch_window = trial_data["pitch"][steady_idx:]
        rms_roll_deg  = np.rad2deg(np.sqrt(np.mean(roll_window**2)))
        rms_pitch_deg = np.rad2deg(np.sqrt(np.mean(pitch_window**2)))
    else:
        rms_roll_deg  = 1000.0
        rms_pitch_deg = 1000.0

    opt_time_sec   = 0.0
    total_time_sec = opt_time_sec + sim_time_sec

    metrics = {
        "optimizer":     optimizer_name,
        "seed":          seed,
        "trial":         trial_idx,
        "J":             J,
        "CoT":           CoT,
        "forwarddistance": forward_distance,
        "lateraldrift":    lateral_drift,
        "meanvx":          mean_vx,
        "fell":            fell,
        "stabilityindex":  stability,
        "rmsrolldeg":      rms_roll_deg,
        "rmspitchdeg":     rms_pitch_deg,
        "opttimesec":      opt_time_sec,
        "simtimesec":      sim_time_sec,
        "totaltimesec":    total_time_sec,
        "couplinggain":   params_np[0],
        "wswing":         params_np[1],
        "wstance":        params_np[2],
        "FFAST":          params_np[3],
        "STOPGAIN":       params_np[4],
        "hipamplitude":   params_np[5],
        "kneeamplitude":  params_np[6],
        "b":              params_np[7],
    }

    return J, metrics


# ============================================================================
# BAYESIAN OPTIMIZATION DRIVER WITH CSV LOGGING
# ============================================================================

def bo_optimize_cpg(
    bounds: torch.Tensor,
    target_forward_position: float,
    robot_mass: float,
    n_trials: int = 100,
    n_init: int = 5,
    optimizer_name: str = "BO",
    seed: int = 0,
    results_dir: str = "results",
    beta_init: float = 5.0,
    beta_min: float = 1.0,
    n_decay_start: int = 40,
    gamma: float = 0.9,
) -> tuple:
    """Bayesian Optimization loop with standardized CSV logging."""
    global quadruped

    if n_init > n_trials:
        print(f"[bo_optimize_cpg] n_init ({n_init}) > n_trials ({n_trials}); capping to {n_trials}.")
        n_init = n_trials

    os.makedirs(results_dir, exist_ok=True)

    # CSV schema must match MARXEFE CSV exactly
    csv_path = os.path.join(results_dir, f"{optimizer_name}_seed{seed}.csv")
    csv_columns = [
        "optimizer", "seed", "trial",
        "J", "CoT",
        "forwarddistance", "lateraldrift", "meanvx",
        "fell", "stabilityindex",
        "rmsrolldeg", "rmspitchdeg",
        "opttimesec", "simtimesec", "totaltimesec",
        "couplinggain", "wswing", "wstance",
        "FFAST", "STOPGAIN",
        "hipamplitude", "kneeamplitude", "b",
    ]
    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    if quadruped is None:
        dt = 0.01
        load_environment(dt)
        quadruped, _, _, _, _ = load_robot(p)
        print(f"\n✅ Environment initialized with robot ID: {quadruped}")

    schedule = BetaSchedule(
        beta_init=beta_init,
        beta_min=beta_min,
        n_decay_start=n_decay_start,
        gamma=gamma,
    )
    bo = BOOptimizer(
        bounds=bounds,
        beta_schedule=schedule,
        n_init=n_init,
        seed=seed,
    )

    objectives       = []
    cots             = []
    forward_distances = []
    lateral_drifts   = []
    mean_velocities  = []
    stabilities      = []
    fall_flags       = []

    print("\n" + "="*70)
    print("BAYESIAN OPTIMIZATION OF CPG PARAMETERS")
    print("="*70)
    print(f"Target forward position: {target_forward_position} m (lateral target = 0)")
    print(f"Robot mass: {robot_mass} kg")
    print(f"Total trials: {n_trials} (init: {n_init}, BO: {n_trials - n_init})")
    print(f"Optimizer: {optimizer_name}, Seed: {seed}")
    print(f"Results: {csv_path}")
    print(f"✅ Reset: MARXEFE-style (neutral pose → standing → CPG)")
    print(f"✅ Distance: FORWARD-ONLY (Y-axis)")
    print(f"✅ Lateral drift: X-axis deviation tracked")
    print("="*70)

    print(f"\nPhase 1: Random Initialization ({n_init} trials)")
    print("-"*70)

    param_names = ["couplinggain", "wswing", "wstance", "FFAST",
                   "STOPGAIN", "hipamplitude", "kneeamplitude", "b"]

    # ------------------------------------------------------------------
    # INITIALIZATION PHASE
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    for i in range(n_init):
        x_np = rng.uniform(bo.lower.numpy(), bo.upper.numpy())

        J, metrics = evaluate_candidate(x_np, target_forward_position, robot_mass,
                                        optimizer_name, seed, i + 1)

        metrics["opttimesec"]   = 0.0
        metrics["totaltimesec"] = metrics["simtimesec"]

        csv_writer.writerow(metrics)
        csv_file.flush()

        bo._append(x_np, J)

        objectives.append(J)
        cots.append(metrics["CoT"])
        forward_distances.append(metrics["forwarddistance"])
        lateral_drifts.append(metrics["lateraldrift"])
        mean_velocities.append(metrics["meanvx"])
        stabilities.append(metrics["stabilityindex"])
        fall_flags.append(metrics["fell"])

        fall_status = "FELL" if metrics["fell"] else "OK"
        print(f"\nTrial {i+1}/{n_trials}:")
        print(f" CPG params: " + ", ".join([f"{n}={v:.3f}" for n, v in zip(param_names, x_np)]))
        print(f" Result: J = {J:.2f}, [{fall_status}], FwdDist = {metrics['forwarddistance']:.3f}m, LatDrift = {metrics['lateraldrift']:.3f}m")
        print(f" CoT = {metrics['CoT']:.3f}, Vel = {metrics['meanvx']:.3f}m/s, Stab = {metrics['stabilityindex']:.2f}°")

        best_idx_so_far = int(bo.train_Y.argmax().item())
        best_J_so_far   = float(bo.train_Y.max().item())
        print(f" Best so far: J = {best_J_so_far:.2f} (trial {best_idx_so_far+1})")

    print(f"\nInitialization complete. Best J = {float(bo.train_Y.max().item()):.2f}")

    print(f"\nPhase 2: Bayesian Optimization ({n_trials - n_init} trials)")
    print("-"*70)

    # ------------------------------------------------------------------
    # BO PHASE
    # ------------------------------------------------------------------
    for t in range(n_init, n_trials):
        opt_start    = time.time()
        model        = bo.fit_model()
        beta_t       = bo.beta_schedule(t)
        x_next_unit  = bo.suggest(model, beta_t)
        x_next_np    = bo.from_unit(x_next_unit)
        opt_time_sec = time.time() - opt_start

        J_next, metrics = evaluate_candidate(x_next_np, target_forward_position, robot_mass,
                                             optimizer_name, seed, t + 1)

        metrics["opttimesec"]   = opt_time_sec
        metrics["totaltimesec"] = opt_time_sec + metrics["simtimesec"]

        csv_writer.writerow(metrics)
        csv_file.flush()

        bo._append(x_next_np, J_next)

        objectives.append(J_next)
        cots.append(metrics["CoT"])
        forward_distances.append(metrics["forwarddistance"])
        lateral_drifts.append(metrics["lateraldrift"])
        mean_velocities.append(metrics["meanvx"])
        stabilities.append(metrics["stabilityindex"])
        fall_flags.append(metrics["fell"])

        best_J   = float(bo.train_Y.max().item())
        best_idx = int(bo.train_Y.argmax().item())
        fall_status = "FELL" if metrics["fell"] else "OK"

        print(f"\nTrial {t+1}/{n_trials}:")
        print(f" CPG params: " + ", ".join([f"{n}={v:.3f}" for n, v in zip(param_names, x_next_np)]))
        print(f" Result: J = {J_next:.2f}, [{fall_status}], FwdDist = {metrics['forwarddistance']:.3f}m, LatDrift = {metrics['lateraldrift']:.3f}m")
        print(f" CoT = {metrics['CoT']:.3f}, Vel = {metrics['meanvx']:.3f}m/s, Stab = {metrics['stabilityindex']:.2f}°")
        print(f" Timing: opt={opt_time_sec:.2f}s, sim={metrics['simtimesec']:.2f}s, total={metrics['totaltimesec']:.2f}s")
        print(f" Best so far: J = {best_J:.2f} (trial {best_idx+1})")

    csv_file.close()
    print(f"\n✅ CSV results saved to: {csv_path}")

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)

    best_params, best_J = bo.best()
    train_X_orig = bo.train_X
    train_Y      = bo.train_Y
    best_idx     = int(train_Y.argmax().item())

    print(f"Best objective: J = {best_J:.2f} (trial {best_idx+1})")
    print(f"\nBest parameters:")
    for name, value in zip(param_names, best_params):
        print(f" {name:16s} = {value:.4f}")
    print("="*70)

    # ------------------------------------------------------------------
    # PLOTTING: THREE COMPREHENSIVE FIGURES
    # ------------------------------------------------------------------
    trials            = np.arange(1, n_trials + 1)
    objectives        = np.array(objectives)
    cots              = np.array(cots)
    forward_distances = np.array(forward_distances)
    lateral_drifts    = np.array(lateral_drifts)
    mean_velocities   = np.array(mean_velocities)
    stabilities       = np.array(stabilities)
    fall_flags        = np.array(fall_flags).astype(int)
    cumulative_falls  = np.cumsum(fall_flags)
    fall_rate         = cumulative_falls / trials * 100

    best_J_so_far = np.maximum.accumulate(objectives)

    J_thresh = 2.0
    r_thresh = 15.0

    condition = (best_J_so_far >= J_thresh) & (fall_rate <= r_thresh)
    if np.any(condition):
        N_walk = int(trials[condition][0])
    else:
        N_walk = int(n_trials)

    D_cum = float(np.sum(forward_distances))

    print(f"\nN_walk (J >= {J_thresh}, fall_rate <= {r_thresh}%): {N_walk}")
    print(f"D_cum over {n_trials} trials: {D_cum:.3f} m")

    # FIGURE 1: Performance Metrics
    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle('Performance Metrics Over Optimization', fontsize=16, fontweight='bold')

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(trials, forward_distances, 'o-', linewidth=2, markersize=4,
             color='steelblue', label='Forward Distance (X-axis)')
    ax1.axhline(y=np.max(forward_distances), color='green', linestyle='--', linewidth=1.5,
                label=f'Max: {np.max(forward_distances):.3f}m', alpha=0.7)
    target_distance = target_forward_position
    ax1.axhline(y=target_distance, color='red', linestyle='--', linewidth=1.5,
            label=f'Target: {target_distance:.2f} m', alpha=0.7)
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Forward Distance [m]', fontsize=11)
    ax1.set_title('Forward Distance (Straight Motion)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(trials, lateral_drifts, 'o-', linewidth=2, markersize=4,
             color='orange', label='Lateral Drift')
    valid_drifts = lateral_drifts[lateral_drifts < 10]
    if len(valid_drifts) > 0:
        ax2.axhline(y=np.min(valid_drifts), color='green', linestyle='--',
                    linewidth=1.5, label=f'Min: {np.min(valid_drifts):.3f}m', alpha=0.7)
    ax2.set_xlabel('Trial Number', fontsize=11)
    ax2.set_ylabel('Lateral Drift [m]', fontsize=11)
    ax2.set_title('Lateral Drift (Y-axis Deviation)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(trials, stabilities, 'o-', linewidth=2, markersize=4,
             color='purple', label='Stability')
    valid_stabilities = stabilities[stabilities < 100]
    if len(valid_stabilities) > 0:
        ax3.axhline(y=np.min(valid_stabilities), color='green', linestyle='--',
                    linewidth=1.5, label=f'Min: {np.min(valid_stabilities):.2f}°', alpha=0.7)
    ax3.set_xlabel('Trial Number', fontsize=11)
    ax3.set_ylabel('Combined RMS Stability [deg]', fontsize=11)
    ax3.set_title('Stability Metric', fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(trials, fall_rate, linewidth=2.5, color='crimson', label='Cumulative Fall Rate')
    ax4.fill_between(trials, 0, fall_rate, alpha=0.2, color='crimson')
    ax4.set_xlabel('Trial Number', fontsize=11)
    ax4.set_ylabel('Fall Rate [%]', fontsize=11)
    ax4.set_title(f'Cumulative Fall Rate (Final: {fall_rate[-1]:.1f}%)', fontsize=12)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig1_path = os.path.join(figures_dir, 'bo_fig1_performance_metrics.png')
    plt.savefig(fig1_path, dpi=150)
    print(f"\n✅ Figure 1 saved as '{fig1_path}'")

    # FIGURE 2: CPG Parameter Evolution
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle('CPG Parameter Evolution', fontsize=14, fontweight='bold')

    param_names_plot = ["Coupling Gain", "w_swing", "w_stance", "F_FAST",
                        "STOP_GAIN", "Hip Amplitude", "Knee Amplitude", "b (Sharpness)"]

    for i, name in enumerate(param_names_plot):
        ax = plt.subplot(3, 3, i+1)
        param_values = train_X_orig[:, i].numpy()
        ax.plot(trials, param_values, 'o-', linewidth=1.5, markersize=4, alpha=0.7)
        ax.axhline(y=best_params[i], color='red', linestyle='--', linewidth=2,
                   label=f'Best: {best_params[i]:.3f}', alpha=0.8)
        ax.axhline(y=bounds[0, i].item(), color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=bounds[1, i].item(), color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Trial', fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, 'bo_fig2_parameter_evolution.png')
    plt.savefig(fig2_path, dpi=150)
    print(f"✅ Figure 2 saved as '{fig2_path}'")

    # FIGURE 3: Objective, CoT, Velocity, Stability
    fig3 = plt.figure(figsize=(16, 12))
    fig3.suptitle('Optimization Metrics: Objective, CoT, Velocity, Stability', fontsize=16, fontweight='bold')

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(trials, objectives, 'o-', linewidth=2, markersize=4, color='purple', label='Objective J')
    ax1.axhline(y=best_J, color='green', linestyle='--', linewidth=2,
                label=f'Best: {best_J:.2f}', alpha=0.8)
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Objective J', fontsize=11)
    ax1.set_title('Objective Function (Paper Formulation)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(4, 1, 2)
    cots_viz = np.clip(cots, 0, 50)
    ax2.plot(trials, cots_viz, 'o-', linewidth=2, markersize=4, color='darkorange', label='CoT')
    valid_cots = cots[cots < 50]
    if len(valid_cots) > 0:
        ax2.axhline(y=np.min(valid_cots), color='green', linestyle='--',
                    linewidth=1.5, label=f'Min: {np.min(valid_cots):.3f}', alpha=0.7)
    ax2.set_xlabel('Trial Number', fontsize=11)
    ax2.set_ylabel('Cost of Transport', fontsize=11)
    ax2.set_title('Cost of Transport (Lower is Better)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(trials, mean_velocities, 'o-', linewidth=2, markersize=4, color='teal', label='Mean Velocity')
    ax3.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5,
                label='Target: 0.8 m/s', alpha=0.7)
    ax3.set_xlabel('Trial Number', fontsize=11)
    ax3.set_ylabel('Mean Velocity [m/s]', fontsize=11)
    ax3.set_title('Mean Forward Velocity', fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(4, 1, 4)
    stab_viz = np.clip(stabilities, 0, 20)
    ax4.plot(trials, stab_viz, 'o-', linewidth=2, markersize=4, color='magenta', label='RMS Stability')
    valid_stab = stabilities[stabilities < 20]
    if len(valid_stab) > 0:
        ax4.axhline(y=np.min(valid_stab), color='green', linestyle='--',
                    linewidth=1.5, label=f'Min: {np.min(valid_stab):.2f}°', alpha=0.7)
    ax4.set_xlabel('Trial Number', fontsize=11)
    ax4.set_ylabel('RMS Stability [deg]', fontsize=11)
    ax4.set_title('Combined RMS Stability (Roll + Pitch)', fontsize=12)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, 'bo_fig3_optimization_metrics.png')
    plt.savefig(fig3_path, dpi=150)
    print(f"✅ Figure 3 saved as '{fig3_path}'")

    return train_X_orig, train_Y, best_params, N_walk, D_cum


if __name__ == "__main__":
    train_X, train_Y, best_params, N_walk, D_cum = bo_optimize_cpg(
        bounds,
        target_forward_position=4.0,
        robot_mass=10.0,
        n_trials=200,
        n_init=5,
        optimizer_name="BO",
        seed=0
    )

    data_dir = "results/data"
    os.makedirs(data_dir, exist_ok=True)
    npz_path = os.path.join(data_dir, "bo_results.npz")
    np.savez(npz_path,
             trainX=train_X.numpy(),
             trainY=train_Y.numpy(),
             bestparams=best_params,
             N_walk=N_walk,
             D_cum=D_cum)
    print(f"N_walk = {N_walk}, D_cum = {D_cum:.3f} m")
    print(f"\n✅ Results saved to '{npz_path}'")
    print("✅ All plots saved!")
