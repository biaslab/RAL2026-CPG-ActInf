"""MARXEFE active-inference optimizer for CPG parameters.

Provides:
  * load_environment, load_robot, reset_simulation, extract_observation,
    check_if_fallen — PyBullet helpers shared with the BO pipeline.
  * run_episode_maxrefe — one MARXEFE-controlled episode (no rendering).
  * compute_objective — same J as methods.bo_optimizer for direct comparison.
  * evaluate_candidate — wrapper returning (J, metrics) with the BO-aligned
    CSV schema.
  * marxefe_optimize_cpg — main optimization loop with CSV logging.
  * plot_marxefe_results — three figures matching the BO ones.

Observation: y_k = [pos_x, pos_y, pitch, roll] (Dy=4).
Goal prior:  m_star = [target_forward_position, 0, 0, 0].
"""

import csv
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pybullet as p
import pybullet_data
import torch
from numpy.linalg import slogdet
from scipy.linalg import det, inv
from scipy.optimize import minimize
from scipy.special import gamma, gammaln
from scipy.stats import multivariate_normal

from methods.cpg_bounds import bounds, bounds_lower, bounds_upper


# =============================================================================
# MARX AGENT — active inference with a Multivariate Auto-Regressive eXogenous
# model. Parameters are inferred via Bayesian filtering, controls via expected
# free energy minimization.
# =============================================================================

class MARXAgent:
    """Active inference agent from Multivariate Auto-Regressive eXogenous model."""

    def __init__(self,
                 coefficients_mean_matrix,
                 coefficients_row_covariance,
                 precision_scale,
                 precision_degrees,
                 control_prior_mean,
                 control_prior_precision,
                 goal_prior,
                 Dy=2,
                 Du=2,
                 delay_inp=1,
                 delay_out=1,
                 time_horizon=1,
                 num_iters=10):

        self.Dy = Dy
        self.Dx = Du * (delay_inp + 1) + Dy * delay_out
        self.Du = Du
        self.ybuffer = np.zeros((Dy, delay_out))            # past outputs
        self.ubuffer = np.zeros((Du, delay_inp + 1))        # current + past inputs
        self.delay_inp = delay_inp
        self.delay_out = delay_out
        self.M = coefficients_mean_matrix                   # A prior/posterior mean
        self.Λ = coefficients_row_covariance                # row precision
        self.Ω = precision_scale                            # precision scale
        self.ν = precision_degrees
        self.μ = control_prior_mean
        self.Υ = control_prior_precision
        self.goal_prior = goal_prior
        self.thorizon = time_horizon
        self.num_iters = num_iters
        self.free_energy = float('inf')

    def update(self, y_k, u_k):
        M0 = self.M
        Λ0 = self.Λ
        Ω0 = self.Ω
        ν0 = self.ν

        self.ubuffer = self.backshift(self.ubuffer, u_k)
        x_k = np.concatenate([self.ubuffer.flatten(), self.ybuffer.flatten()])

        X = np.outer(x_k, x_k)
        Ξ = np.outer(x_k, y_k) + np.dot(Λ0, M0)

        self.ν = ν0 + 1
        self.Λ = Λ0 + X
        self.Ω = Ω0 + np.outer(y_k, y_k) + np.dot(M0.T, np.dot(Λ0, M0)) - np.dot(Ξ.T, np.dot(inv(Λ0 + X), Ξ))
        self.M = np.dot(inv(Λ0 + X), Ξ)

        self.ybuffer = self.backshift(self.ybuffer, y_k)
        return None

    def params(self):
        return self.M, self.U, self.V, self.ν

    def log_evidence(self, y, x):
        η, μ, Ψ = self.posterior_predictive(x)
        return -0.5 * (self.Dy * np.log(η * np.pi) - np.log(det(Ψ)) - 2 * self.logmultigamma(self.Dy, (η + self.Dy) / 2) +
                       2 * self.logmultigamma(self.Dy, (η + self.Dy - 1) / 2) + (η + self.Dy) * np.log(1 + 1 / η * np.dot((y - μ).T, np.dot(Ψ, (y - μ)))))

    def posterior_predictive(self, x_t):
        """Student-t predictive distribution."""
        η_t = self.ν - self.Dy + 1
        μ_t = np.dot(self.M.T, x_t)
        Ψ_t = (self.ν - self.Dy + 1) * inv(self.Ω) / (1 + np.dot(x_t, np.dot(inv(self.Λ), x_t)))
        return η_t, μ_t, Ψ_t

    def predictions(self, controls, time_horizon=1):
        m_y = np.zeros((self.Dy, time_horizon))
        S_y = np.zeros((self.Dy, self.Dy, time_horizon))

        ybuffer = self.ybuffer
        ubuffer = self.ubuffer

        for t in range(time_horizon):
            ubuffer = self.backshift(ubuffer, controls[:, t])
            x_t = np.concatenate([ubuffer.flatten(), ybuffer.flatten()])

            η_t, μ_t, Ψ_t = self.posterior_predictive(x_t)
            m_y[:, t] = μ_t
            S_y[:, :, t] = inv(Ψ_t) * η_t / (η_t - 2)

            ybuffer = self.backshift(ybuffer, m_y[:, t])

        return m_y, S_y

    def mutualinfo(self, x):
        _, _, Ψ = self.posterior_predictive(x)
        _, logdet = slogdet(Ψ)
        return logdet

    def crossentropy(self, x):
        m_star = self.goal_prior.mean
        S_star = self.goal_prior.cov
        η_t, μ_t, Ψ_t = self.posterior_predictive(x)
        return 0.5 * (η_t / (η_t - 2) * np.trace(np.dot(inv(S_star), inv(Ψ_t))) + np.dot((μ_t - m_star).T, np.dot(inv(S_star), (μ_t - m_star))))

    def EFE(self, controls):
        ybuffer = self.ybuffer
        ubuffer = self.ubuffer

        J = 0
        for t in range(self.thorizon):
            u_t = controls[t * self.Du:(t + 1) * self.Du]
            ubuffer = self.backshift(ubuffer, u_t)
            x_t = np.concatenate([ubuffer.flatten(), ybuffer.flatten()])

            J += self.mutualinfo(x_t) + self.crossentropy(x_t) + np.dot(u_t - self.μ, np.dot(self.Υ, u_t - self.μ)) / 2.0

            _, m_y, _ = self.posterior_predictive(x_t)
            ybuffer = self.backshift(ybuffer, m_y)

        return J

    def minimizeEFE(self, u_0=None, verbose=False, control_lims=(-np.inf, np.inf)):
        if u_0 is None:
            u_0 = 1e-8 * np.random.randn(self.thorizon)

        def J(u):
            return self.EFE(u)

        bounds = [control_lims] * u_0.size
        result = minimize(J,
                          u_0,
                          method='L-BFGS-B',
                          bounds=bounds,
                          options={'disp': verbose, 'maxiter': 10000})
        return result.x

    def backshift(self, x, a):
        if x.ndim == 2:
            return np.column_stack((a, x[:, :-1]))
        elif x.ndim == 1:
            N = x.size
            S = np.eye(N, k=-1)
            e = np.zeros(N)
            e[0] = 1.0
            return S.dot(x) + e * a

    def update_goals(self, x, g):
        x = np.roll(x, -1)
        x[-1] = g
        return x

    def multigamma(self, p, a):
        result = np.pi ** (p * (p - 1) / 4)
        for j in range(1, p + 1):
            result *= gamma(a + (1 - j) / 2)
        return result

    def logmultigamma(self, p, a):
        result = p * (p - 1) / 4 * np.log(np.pi)
        for j in range(1, p + 1):
            result += gammaln(a + (1 - j) / 2)
        return result

    def save_agent(self, filename, makedir=False):
        if not filename.endswith(".pkl"):
            print("Filename does not end with .pkl. Appending .pkl to filename")
            filename += ".pkl"

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            if makedir:
                print(f"Directory {directory} does not exist. Creating directory")
                os.makedirs(directory)
            else:
                raise FileNotFoundError(f"Directory {directory} does not exist. Cannot save to {filename}")

        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Agent saved to {filename}")

    def load_agent(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError("File does not exist.")

        with open(filename, 'rb') as file:
            return pickle.load(file)

    def reset_buffer(self):
        self.ubuffer = np.zeros((self.Du, self.delay_inp + 1))
        self.ybuffer = np.zeros((self.Dy, self.delay_out))
        self.free_energy = float('inf')


def acc2pos(acc, prev_state, dt=1.0, reg=1e-3):
    """Kalman filter for accelerometer integration."""
    A = np.array([[1, 0, 0, dt,  0,  0, dt**2/2,       0,       0],
                  [0, 1, 0,  0, dt,  0,       0, dt**2/2,       0],
                  [0, 0, 1,  0,  0, dt,       0,       0, dt**2/2],
                  [0, 0, 0,  1,  0,  0,      dt,       0,       0],
                  [0, 0, 0,  0,  1,  0,       0,      dt,       0],
                  [0, 0, 0,  0,  0,  1,       0,       0,      dt],
                  [0, 0, 0,  0,  0,  0,       1,       0,       0],
                  [0, 0, 0,  0,  0,  0,       0,       1,       0],
                  [0, 0, 0,  0,  0,  0,       0,       0,       1]])

    σ = 1.0
    block1 = np.diag(np.repeat([dt**5/20], 3))
    block2 = np.diag(np.repeat([dt**4/8], 3))
    block3 = np.diag(np.repeat([dt**3/6], 3))
    block4 = np.diag(np.repeat([dt**2/2], 3))
    block5 = np.diag(np.repeat([dt], 3))
    Q = σ * np.block([[block1, block2, block3],
                      [block2, block3, block4],
                      [block3, block4, block5]])

    C = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    ρ = 1.0
    R = np.diag(ρ * np.ones(3))

    state_pred_m = A @ prev_state.mean
    state_pred_S = A @ prev_state.cov @ A.T + Q

    Is = C @ state_pred_S @ C.T + R
    Kg = state_pred_S @ C.T @ inv(Is)
    state_m = state_pred_m + Kg @ (acc - C @ state_pred_m)
    state_S = (np.eye(9) - Kg @ C) @ state_pred_S + reg * np.eye(9)

    state = multivariate_normal(state_m, state_S)

    return state_m[:3], state

# Control prior hyperparameter: bounds [lower_i, upper_i] correspond to
# mu_i ± n_sigma * sigma_i.
n_sigma = 2.0

# CPG phase classification thresholds for feedback mode switching.
SWING_ENTER, SWING_EXIT, STANCE_ENTER, STANCE_EXIT = 0.15, 0.02, -0.15, -0.02
phase_state_memory = []
quadruped          = None
joint_IDs_full     = None
filtered_joint_IDs = None
feet_joint_IDs     = None


def get_phase(y_val, leg_idx,
              swing_enter=SWING_ENTER,
              swing_exit=SWING_EXIT,
              stance_enter=STANCE_ENTER,
              stance_exit=STANCE_EXIT):
    """Classify CPG phase from y value with hysteresis.
    Uses global phase_state_memory to maintain state per leg.
    """
    global phase_state_memory
    current_state = phase_state_memory[leg_idx]
    if current_state == 'swing':
        new_state = 'transition' if y_val < swing_exit else 'swing'
    elif current_state == 'stance':
        new_state = 'transition' if y_val > stance_exit else 'stance'
    else:
        if y_val > swing_enter:
            new_state = 'swing'
        elif y_val < stance_enter:
            new_state = 'stance'
        else:
            new_state = 'transition'
    phase_state_memory[leg_idx] = new_state
    return new_state


def compute_feedback_u(x_vec, y_vec, w_vec, k_matrix, contacts, phases,
                       F_fast, contact_touch, contact_unload,
                       coupling_gain, STOP_gain):
    """Righetti-style STOP/FAST feedback. Same structure as the BO pipeline."""
    n_legs    = len(x_vec)
    u_fb      = np.zeros(n_legs)
    coupling_y = coupling_gain * (k_matrix @ y_vec)
    modes      = []
    for j in range(n_legs):
        yj      = y_vec[j]
        xj      = x_vec[j]
        wj      = w_vec[j]
        contact = contacts[j]
        phase   = phases[j]
        if phase == 'swing':
            if contact < contact_touch:
                u_fb[j] = STOP_gain * (wj * xj - coupling_y[j])
                modes.append("STOP")
            else:
                u_fb[j] = np.sign(yj) * F_fast
                modes.append("FAST")
        elif phase == 'stance':
            if contact > contact_unload:
                u_fb[j] = STOP_gain * (wj * xj - coupling_y[j])
                modes.append("STOP")
            else:
                u_fb[j] = np.sign(yj) * F_fast
                modes.append("FAST")
        else:
            u_fb[j] = 0.0
            modes.append("NORMAL")
    return u_fb, modes


# =============================================================================
# ENVIRONMENT AND ROBOT SETUP - MATCHES BO
# =============================================================================

def load_environment(dt, use_gui=False):
    """Initialize PyBullet physics engine."""
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
    """Load Laikago robot and extract joint/link information."""
    start_position    = [0.0, 0.0, 0.55]
    start_orientation = [0.0, 0.5, 0.5, 0.0]
    urdfFlags         = p.URDF_USE_SELF_COLLISION

    robot = p.loadURDF(
        "laikago/laikago_toes.urdf",
        start_position,
        start_orientation,
        flags=urdfFlags,
        useFixedBase=False
    )
    p.changeDynamics(robot, -1, mass=10.0)

    n_joints    = p.getNumJoints(robot)
    joints_info = {}
    lower_legs  = []

    for i in range(n_joints):
        info            = p.getJointInfo(robot, i)
        joint_ID        = info[0]
        joint_name      = info[1].decode('utf-8')
        joint_type      = info[2]
        joint_link_name = info[12].decode('utf-8')
        joint_parent_ID = info[16]
        joints_info[joint_ID] = {
            'joint_name':      joint_name,
            'joint_type':      joint_type,
            'joint_link_name': joint_link_name,
            'joint_parent_ID': joint_parent_ID
        }
        if 'lower_leg' in joint_link_name:
            lower_legs.append(joint_ID)

    for l0 in lower_legs:
        for l1 in lower_legs:
            if l0 != l1:
                p.setCollisionFilterPair(robot, robot, l0, l1, 1)

    joint_IDs_full = {
        'FL': [4, 5, 6],
        'FR': [0, 1, 2],
        'RL': [12, 13, 14],
        'RR': [8, 9, 10]
    }
    filtered_joint_IDs = [5, 6, 1, 2, 13, 14, 9, 10]
    feet_joint_IDs     = [7, 3, 15, 11]

    return robot, joints_info, joint_IDs_full, filtered_joint_IDs, feet_joint_IDs


# =============================================================================
# STATE OBSERVATION / FALL DETECTION - MATCHES BO
# =============================================================================

DEFAULT_ORI = [0.0, 0.5, 0.5, 0.0]


def check_if_fallen(p, robot, base_position, base_orientation,
                    fallen_threshold_orientation=0.3,
                    fallen_threshold_height=0.25):
    """Decide whether the robot has fallen (orientation + height criteria)."""
    rot_mat       = p.getMatrixFromQuaternion(base_orientation)
    local_up      = np.array(rot_mat[6:])
    world_up      = np.array([0, 0, 1])
    magic_value   = np.dot(world_up, local_up)
    fallen_reason_1 = magic_value < fallen_threshold_orientation
    fallen_reason_2 = base_position[2] < fallen_threshold_height
    is_fallen = fallen_reason_1 or fallen_reason_2
    return is_fallen, magic_value, fallen_reason_1, fallen_reason_2


def get_base_orientation(p, robot, ori_default):
    """Get base orientation relative to the robot's default quaternion."""
    base_pos, base_quat = p.getBasePositionAndOrientation(robot)
    _, base_orientation = p.multiplyTransforms(
        positionA=[0, 0, 0], orientationA=base_quat,
        positionB=[0, 0, 0], orientationB=ori_default
    )
    return base_pos, base_orientation


def extract_observation(p, robot, ori_default):
    """Observation y_k = [pos_x, pos_y, pitch, roll] for MARXEFE (Dy=4).
    pos_x = base_pos[1] (forward +Y); pos_y = base_pos[0] (lateral X).
    """
    base_pos, base_orientation = get_base_orientation(p, robot, ori_default)
    pos_x = base_pos[1]   # forward +Y
    pos_y = base_pos[0]   # lateral  X
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation)
    y_k = np.array([pos_x, pos_y, pitch, roll])
    return y_k, base_pos, base_orientation


def reset_simulation(p, robot, filtered_joint_IDs, ori_default):
    """Reset robot to neutral standing pose and run pre-trial settling."""
    start_position = [0.0, 0.0, 0.55]
    p.resetBasePositionAndOrientation(robot, start_position, ori_default)
    p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])

    abduction_joint_ids = [0, 4, 8, 12]
    hip_joint_ids       = [1, 5, 9, 13]
    knee_joint_ids      = [2, 6, 10, 14]

    for jid in abduction_joint_ids:
        p.resetJointState(robot, jid,  0.0)
    for jid in hip_joint_ids:
        p.resetJointState(robot, jid,  0.05)
    for jid in knee_joint_ids:
        p.resetJointState(robot, jid, -0.6)

    # Phase 1: PRE-TRIAL SETTLING 1.0 s = 100 steps
    for _ in range(100):
        for jid in abduction_joint_ids:
            p.setJointMotorControl2(
                robot, jid, p.POSITION_CONTROL,
                targetPosition=0.0, force=500
            )
        for jid in hip_joint_ids:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, 0.25)
        for jid in knee_joint_ids:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, -1.0)
        p.stepSimulation()

    # Initialize CPG on limit cycle
    n_legs = 4
    u      = 2.0
    cpg_x  = np.zeros(n_legs)
    cpg_y  = np.zeros(n_legs)
    theta  = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    for i in range(n_legs):
        cpg_x[i] = np.sqrt(u) * np.cos(theta[i])
        cpg_y[i] = np.sqrt(u) * np.sin(theta[i])
    return cpg_x, cpg_y


# =============================================================================
# MARXEFE EPISODE
# =============================================================================

def run_episode_maxrefe(agent, robot, joint_IDs_full_arg,
                        filtered_joint_IDs_arg, feet_joint_IDs_arg,
                        dt, episode_length=4.5,
                        lambda_energy=1e-2,
                        target_forward_position=4.0):
    """Run one MARXEFE-controlled episode.
    Trial structure: Phase1(100 steps settle, in reset) + Phase2(150 transition)
    + Phase3(300 steady) = 4.5 s at dt=0.01.
    """
    ori_default = [0.0, 0.5, 0.5, 0.0]

    cpg_x, cpg_y = reset_simulation(p, robot, filtered_joint_IDs_arg, ori_default)

    n_legs = 4
    y_k, base_pos, base_orientation = extract_observation(p, robot, ori_default)

    global phase_state_memory
    phase_state_memory = []
    for j in range(n_legs):
        if   cpg_y[j] > SWING_ENTER:
            phase_state_memory.append('swing')
        elif cpg_y[j] < STANCE_ENTER:
            phase_state_memory.append('stance')
        else:
            phase_state_memory.append('transition')

    coupling_gain  = agent.μ[0]
    w_swing        = agent.μ[1]
    w_stance       = agent.μ[2]
    F_FAST         = agent.μ[3]
    STOP_GAIN      = agent.μ[4]
    hip_amplitude  = agent.μ[5]
    knee_amplitude = agent.μ[6]
    b              = agent.μ[7]

    alpha      = 3.0
    beta       = 12.0
    u          = 2.0
    hip_offset  = 0.26
    knee_offset = -1.0

    contact_touch  = 0.5
    contact_unload = 0.5
    k_matrix = np.array([
        [ 0, -1, -1,  1],
        [-1,  0,  1, -1],
        [-1,  1,  0, -1],
        [ 1, -1, -1,  0]
    ], dtype=float)

    action_update_frequency = 50

    transition_duration = 1.5
    trial_duration      = episode_length
    num_steps           = int(trial_duration / dt)
    transition_steps    = int(transition_duration / dt)

    times       = np.zeros(num_steps)
    y_history   = np.zeros((4, num_steps))
    u_history   = np.zeros((8, num_steps))
    preds_m     = np.zeros((4, num_steps))
    preds_v     = np.zeros((4, num_steps))
    goals_m     = np.zeros((4, num_steps))
    positions   = []

    roll_angles  = np.zeros(num_steps)
    pitch_angles = np.zeros(num_steps)

    n_joints    = 2 * n_legs
    torques_log = np.zeros((n_joints, num_steps))
    qdot_log    = np.zeros((n_joints, num_steps))
    base_pos_log = np.zeros((num_steps, 3))
    vx_log      = np.zeros(num_steps)
    vy_log      = np.zeros(num_steps)

    policy = np.tile(agent.μ, agent.thorizon) + 0.05 * rnd.randn(agent.Du * agent.thorizon)
    is_fallen = False

    leg_names      = ["FL", "FR", "RL", "RR"]
    hip_joint_ids  = [1, 5, 9, 13]
    knee_joint_ids = [2, 6, 10, 14]

    def EFE_with_energy(policy_flat):
        G_original  = agent.EFE(policy_flat)
        effort_term = 0.0
        for t_h in range(agent.thorizon):
            u_t = policy_flat[t_h * agent.Du:(t_h + 1) * agent.Du]
            effort_term += np.sum(u_t ** 2)
        return G_original + lambda_energy * effort_term

    DEBOUNCE_THRESHOLD   = 2
    debounced_contacts   = np.zeros(n_legs, dtype=int)
    contact_change_count = np.zeros(n_legs, dtype=int)
    for k_step in range(num_steps):
        t = k_step * dt
        times[k_step]      = t
        y_history[:, k_step] = y_k
        goals_m[:, k_step]   = agent.goal_prior.mean

        x_k              = np.concatenate([agent.ubuffer.flatten(),
                                            agent.ybuffer.flatten()])
        eta_k, mu_k, Psi_k = agent.posterior_predictive(x_k)
        preds_m[:, k_step]  = mu_k
        preds_v[:, k_step]  = np.diag(inv(Psi_k) * eta_k / (eta_k - 2))

        params_8d = np.array([
            coupling_gain, w_swing, w_stance, F_FAST,
            STOP_GAIN, hip_amplitude, knee_amplitude, b
        ])
        params_8d = np.clip(params_8d, bounds_lower.numpy(), bounds_upper.numpy())
        coupling_gain  = params_8d[0]
        w_swing        = params_8d[1]
        w_stance       = params_8d[2]
        F_FAST         = params_8d[3]
        STOP_GAIN      = params_8d[4]
        hip_amplitude  = params_8d[5]
        knee_amplitude = params_8d[6]
        b              = params_8d[7]

        w_vec = np.zeros(n_legs)
        r_vec = np.zeros(n_legs)
        for j in range(n_legs):
            y_prev = cpg_y[j]
            x_prev = cpg_x[j]
            w = (w_stance / (np.exp(-b * y_prev) + 1) +
                 w_swing  / (np.exp( b * y_prev) + 1))
            w_vec[j] = w
            r        = np.sqrt(x_prev ** 2 + y_prev ** 2)
            r_vec[j] = r
            cpg_x[j] += dt * (alpha * (u - r ** 2) * x_prev - w * y_prev)

        raw_contacts = np.array([
            int(len(p.getContactPoints(
                bodyA=0, bodyB=robot,
                linkIndexA=-1, linkIndexB=ID)) > 0)
            for ID in feet_joint_IDs_arg
        ])
        for j in range(n_legs):
            if raw_contacts[j] == debounced_contacts[j]:
                contact_change_count[j] = 0
            else:
                contact_change_count[j] += 1
                if contact_change_count[j] >= DEBOUNCE_THRESHOLD:
                    debounced_contacts[j] = raw_contacts[j]
                    contact_change_count[j] = 0
        phases = [get_phase(cpg_y[j], j) for j in range(n_legs)]

        u_fb, modes = compute_feedback_u(
            x_vec=cpg_x, y_vec=cpg_y, w_vec=w_vec, k_matrix=k_matrix,
            contacts=debounced_contacts, phases=phases, F_fast=F_FAST,
            contact_touch=contact_touch, contact_unload=contact_unload,
            coupling_gain=coupling_gain, STOP_gain=STOP_GAIN
        )

        for j in range(n_legs):
            y_prev        = cpg_y[j]
            x_prev        = cpg_x[j]
            r             = r_vec[j]
            w             = w_vec[j]
            coupling_term = coupling_gain * np.dot(k_matrix[j, :], cpg_y)
            cpg_y[j] += dt * (beta * (u - r ** 2) * y_prev +
                               w * x_prev + coupling_term + u_fb[j])

        for j in range(n_legs):
            leg_name                       = leg_names[j]
            abd_joint, hip_joint, kn_joint = joint_IDs_full_arg[leg_name]
            p.setJointMotorControl2(
                robot, abd_joint, p.POSITION_CONTROL,
                targetPosition=0.0, force=500
            )
            hip_angle  = hip_offset  + hip_amplitude  * cpg_x[j]
            knee_angle = knee_offset - knee_amplitude  * max(0, cpg_y[j])
            p.setJointMotorControl2(robot, hip_joint, p.POSITION_CONTROL, hip_angle)
            p.setJointMotorControl2(robot, kn_joint,  p.POSITION_CONTROL, knee_angle)

        p.stepSimulation()

        joint_idx = 0
        for jid_hip, jid_knee in zip(hip_joint_ids, knee_joint_ids):
            hs = p.getJointState(robot, jid_hip)
            ks = p.getJointState(robot, jid_knee)
            torques_log[joint_idx,   k_step] = hs[3]
            torques_log[joint_idx+1, k_step] = ks[3]
            qdot_log[joint_idx,      k_step] = hs[1]
            qdot_log[joint_idx+1,    k_step] = ks[1]
            joint_idx += 2

        y_k_new, base_pos, base_orientation = extract_observation(
            p, robot, ori_default
        )
        positions.append(np.array(base_pos))
        base_pos_log[k_step, :] = base_pos

        base_vel, _ = p.getBaseVelocity(robot)
        vx_log[k_step] = base_vel[1]   # forward +Y
        vy_log[k_step] = base_vel[0]   # lateral  X

        roll, pitch, _ = p.getEulerFromQuaternion(base_orientation)
        roll_angles[k_step]  = roll
        pitch_angles[k_step] = pitch

        agent.update(y_k_new, np.array([
            coupling_gain, w_swing, w_stance, F_FAST,
            STOP_GAIN, hip_amplitude, knee_amplitude, b
        ]))
        u_history[:, k_step] = [
            coupling_gain, w_swing, w_stance, F_FAST,
            STOP_GAIN, hip_amplitude, knee_amplitude, b
        ]

        if k_step > 0 and k_step % action_update_frequency == 0:
            bounds_agent = []
            for _ in range(agent.thorizon):
                for i in range(8):
                    bounds_agent.append((bounds_lower[i].item(), bounds_upper[i].item()))

            try:
                results = minimize(
                    EFE_with_energy,
                    policy,
                    method='L-BFGS-B',
                    bounds=bounds_agent,
                    options={'disp': False, 'maxiter': 100, 'ftol': 1e-4}
                )

                if k_step < 2 * action_update_frequency:
                    policy = np.tile(agent.μ, agent.thorizon)
                else:
                    policy = results.x

            except Exception as e:
                print(f"Warning: EFE minimization failed at step {k_step}: {e}")
                policy = np.tile(agent.μ, agent.thorizon)

            u_t = policy[0:agent.Du]

            coupling_gain  = u_t[0]
            w_swing        = u_t[1]
            w_stance       = u_t[2]
            F_FAST         = u_t[3]
            STOP_GAIN      = u_t[4]
            hip_amplitude  = u_t[5]
            knee_amplitude = u_t[6]
            b              = u_t[7]

        y_k = y_k_new

        is_fallen, _, _, _ = check_if_fallen(
            p, robot, base_pos, base_orientation, 0.3, 0.25
        )
        if is_fallen:
            actual_steps  = k_step + 1
            times         = times[:actual_steps]
            torques_log   = torques_log[:, :actual_steps]
            qdot_log      = qdot_log[:, :actual_steps]
            base_pos_log  = base_pos_log[:actual_steps, :]
            roll_angles   = roll_angles[:actual_steps]
            pitch_angles  = pitch_angles[:actual_steps]
            vx_log        = vx_log[:actual_steps]
            vy_log        = vy_log[:actual_steps]
            num_steps     = actual_steps
            break

    # ------------------------------------------------------------------
    # POST-EPISODE METRICS — steady-state window only
    # ------------------------------------------------------------------
    start_idx = transition_steps

    distance = np.linalg.norm(base_pos_log[-1, :2] - base_pos_log[0, :2])

    if len(roll_angles) > start_idx:
        roll_w   = roll_angles[start_idx:]
        pitch_w  = pitch_angles[start_idx:]
        rms_roll  = np.rad2deg(np.sqrt(np.mean(roll_w  ** 2)))
        rms_pitch = np.rad2deg(np.sqrt(np.mean(pitch_w ** 2)))
        combined_stability = np.sqrt(rms_roll ** 2 + rms_pitch ** 2)
    else:
        combined_stability = 1000.0

    if len(base_pos_log) > 0:
        forward_distance = base_pos_log[-1, 1] - base_pos_log[0, 1]
        lateral_drift    = abs(base_pos_log[-1, 0] - base_pos_log[0, 0])
        T_steady  = (len(base_pos_log) - start_idx) * dt
        mean_vx   = (base_pos_log[-1, 1] - base_pos_log[start_idx, 1]) / T_steady if T_steady > 0 else 0.0
        torques_s = torques_log[:, start_idx:].T
        qdot_s    = qdot_log[:,   start_idx:].T
        mech_pwr  = np.sum(np.abs(torques_s * qdot_s)) * dt
        fwd_steady = base_pos_log[-1, 1] - base_pos_log[start_idx, 1]
        CoT = (mech_pwr / (10.0 * 9.81 * fwd_steady)
               if fwd_steady > 0.001 else 1000.0)
        CoT = min(CoT, 200.0)
    else:
        forward_distance = 0.0
        lateral_drift    = 0.0
        mean_vx          = 0.0
        CoT              = 1000.0

    params_8d = np.array([
        coupling_gain, w_swing, w_stance, F_FAST,
        STOP_GAIN, hip_amplitude, knee_amplitude, b
    ])

    trial_data = {
        "t":                   times,
        "pos_x":               base_pos_log[:, 1],   # forward +Y
        "pos_y":               base_pos_log[:, 0],   # lateral  X
        "vx":                  vx_log,
        "vy":                  vy_log,
        "roll":                roll_angles,
        "pitch":               pitch_angles,
        "yaw":                 np.zeros_like(times),
        "forces":              np.zeros((len(times), n_legs)),
        "torques":             torques_log.T,
        "qdot":                qdot_log.T,
        "base_pos":            base_pos_log,
        "fall":                is_fallen,
        "transition_duration": transition_duration,
        "distance":            distance,
        "forward_distance":    forward_distance,
        "lateral_drift":       lateral_drift,
        "stability":           combined_stability,
        "mean_vx":             mean_vx,
        "CoT":                 CoT,
        "x_cpg":               np.zeros((n_legs, len(times))),
        "y_cpg":               np.zeros((n_legs, len(times))),
        "final_params_8d":     params_8d,
    }
    return trial_data


# =============================================================================
# OBJECTIVE FUNCTION — identical to methods.bo_optimizer.compute_objective
# =============================================================================

def compute_objective(trial_data: dict,
                      target_forward_position: float,
                      robot_mass: float,
                      g: float = 9.81) -> float:
    """Compute J = position_reward - w2*CoT_norm - w3*stability."""
    t                   = trial_data["t"]
    transition_duration = trial_data["transition_duration"]
    steady_idx          = np.searchsorted(t, transition_duration)

    if steady_idx >= len(t) - 5:
        return -50.0

    pos_x_steady   = trial_data["pos_x"][steady_idx:]
    pos_y_steady   = trial_data["pos_y"][steady_idx:]
    torques_steady = trial_data["torques"][steady_idx:, :]
    qdot_steady    = trial_data["qdot"][steady_idx:, :]
    bpos_steady    = trial_data["base_pos"][steady_idx:, :]
    T              = len(pos_x_steady)

    w1          = 5.0
    l_r         = 0.85
    sigma_pos_x = 1.0
    sigma_pos_y = 0.1
    R_p_sum     = 0.0
    for i in range(T):
        err_sq = ((pos_x_steady[i] - target_forward_position) / sigma_pos_x) ** 2 \
               + (pos_y_steady[i] / sigma_pos_y) ** 2
        reward_i = np.exp(-0.5 * err_sq)
        R_p_sum += min(reward_i, l_r)
    R_p           = R_p_sum / T
    position_term = w1 * R_p

    w2          = 0.4
    delta_t     = 0.02
    mech_power  = np.sum(np.abs(torques_steady * qdot_steady)) * delta_t
    d           = bpos_steady[-1, 1] - bpos_steady[0, 1]
    CoT_cap     = 200.0 if d < 0.5 else (150.0 if d < 1.5 else 100.0)
    CoT         = (mech_power / (robot_mass * g * max(d, 0.001))
                   if d < 0.005
                   else mech_power / (robot_mass * g * d))
    CoT         = min(CoT, CoT_cap)
    CoT_norm    = CoT / 50.0

    w3        = 0.02
    stability = trial_data["stability"]

    J = position_term - w2 * CoT_norm - w3 * stability
    return J


# =============================================================================
# EVALUATE_CANDIDATE — MARXEFE wrapper, BO-aligned metrics schema
# =============================================================================

def evaluate_candidate(params_np, target_forward_position, robot_mass,
                       optimizer_name, seed, trial_idx, agent=None):
    """Evaluate one MARXEFE episode and return (J, metrics_dict).
    params_np is unused (agent drives action selection internally).
    """
    sim_start = time.time()
    if agent is None:
        raise NotImplementedError("evaluate_candidate requires an agent.")

    trial_data = run_episode_maxrefe(
        agent, quadruped, joint_IDs_full, filtered_joint_IDs, feet_joint_IDs,
        dt=0.01, episode_length=4.5,
        lambda_energy=1e-2,
        target_forward_position=target_forward_position
    )
    sim_time_sec = time.time() - sim_start

    J = compute_objective(trial_data, target_forward_position, robot_mass)

    forwarddistance = trial_data["forward_distance"]
    lateraldrift    = trial_data["lateral_drift"]
    meanvx          = trial_data["mean_vx"]
    stability       = trial_data["stability"]
    CoT             = trial_data["CoT"]
    fell            = int(trial_data["fall"])

    t_arr               = trial_data["t"]
    transition_duration = trial_data["transition_duration"]
    steady_idx          = np.searchsorted(t_arr, transition_duration)
    if len(trial_data["roll"]) > steady_idx:
        rmsrolldeg  = np.rad2deg(np.sqrt(np.mean(trial_data["roll"][steady_idx:]  ** 2)))
        rmspitchdeg = np.rad2deg(np.sqrt(np.mean(trial_data["pitch"][steady_idx:] ** 2)))
    else:
        rmsrolldeg  = 1000.0
        rmspitchdeg = 1000.0

    opt_time_sec   = 0.0
    total_time_sec = opt_time_sec + sim_time_sec

    params8d = trial_data["final_params_8d"]
    stabilityindex = stability

    metrics = {
        "optimizer":      optimizer_name,
        "seed":           seed,
        "trial":          trial_idx,
        "J":              J,
        "CoT":            CoT,
        "forwarddistance": forwarddistance,
        "lateraldrift":   lateraldrift,
        "meanvx":         meanvx,
        "fell":           fell,
        "stabilityindex": stabilityindex,
        "rmsrolldeg":     rmsrolldeg,
        "rmspitchdeg":    rmspitchdeg,
        "opttimesec":     opt_time_sec,
        "simtimesec":     sim_time_sec,
        "totaltimesec":   total_time_sec,
        "couplinggain":   params8d[0],
        "wswing":         params8d[1],
        "wstance":        params8d[2],
        "FFAST":          params8d[3],
        "STOPGAIN":       params8d[4],
        "hipamplitude":   params8d[5],
        "kneeamplitude":  params8d[6],
        "b":              params8d[7],
    }
    return J, metrics


# =============================================================================
# MARXEFE OPTIMISATION LOOP
# =============================================================================

def marxefe_optimize_cpg(bounds: torch.Tensor,
                         target_forward_position: float,
                         robot_mass: float,
                         n_trials: int = 200,
                         optimizer_name: str = "MARXEFE",
                         seed: int = 0,
                         results_dir: str = "results") -> tuple:
    """MARXEFE optimisation loop with BO-compatible CSV logging."""
    global quadruped, joint_IDs_full, filtered_joint_IDs, feet_joint_IDs

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{optimizer_name}_seed{seed}.csv")

    # CSV schema must match BO exactly (column order matters)
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
        load_environment(dt, use_gui=False)
        (quadruped,
         _,
         joint_IDs_full,
         filtered_joint_IDs,
         feet_joint_IDs) = load_robot(p)
        print(f"\n✅ Environment initialized with robot ID: {quadruped}")

    # MARXEFE agent (internal objective unchanged)
    Mu  = 2
    My  = 3
    Dy  = 4   # observation dim: pos_x, pos_y, pitch, roll
    Du  = 8
    len_horizon = 3

    Nu0     = 20.0
    Omega0  = 1e0 * np.diag(np.ones(Dy))
    reg_dim = Dy * My + Du * (Mu + 1)
    Lambda0 = 1e-3 * np.diag(np.ones(reg_dim))
    Mean0   = 1e-8 * rnd.randn(reg_dim, Dy)

    mu_t      = 0.5 * (bounds_lower + bounds_upper)
    sigma_t   = (bounds_upper - bounds_lower) / (2.0 * n_sigma)
    upsilon_t = 1.0 / (sigma_t ** 2)
    Lambda_u  = torch.diag(upsilon_t)
    mu0       = mu_t.numpy()
    Upsilon0  = Lambda_u.numpy()

    # Goal prior — 4D: [pos_x, pos_y, pitch, roll]
    m_star      = np.array([target_forward_position, 0.0, 0.0, 0.0])
    sigma_pos_x = 1.0
    sigma_pos_y = 0.1
    sigma_pitch = np.deg2rad(5)
    sigma_roll  = np.deg2rad(5)
    v_star      = np.diag([sigma_pos_x**2, sigma_pos_y**2, sigma_pitch**2, sigma_roll**2])
    goal        = multivariate_normal(m_star, v_star)

    agent = MARXAgent(
        coefficients_mean_matrix    = Mean0.copy(),
        coefficients_row_covariance = Lambda0.copy(),
        precision_scale             = Omega0.copy(),
        precision_degrees           = Nu0,
        control_prior_mean          = mu0.copy(),
        control_prior_precision     = Upsilon0.copy(),
        goal_prior                  = goal,
        Dy=Dy, Du=Du, delay_inp=Mu, delay_out=My, time_horizon=len_horizon
    )

    dtype        = torch.double
    train_X_orig = torch.empty(0, bounds.shape[1], dtype=dtype)
    train_Y      = torch.empty(0, 1, dtype=dtype)

    param_names = ["couplinggain", "wswing", "wstance", "FFAST",
                   "STOPGAIN", "hipamplitude", "kneeamplitude", "b"]

    print("\n" + "=" * 70)
    print("MARXEFE OPTIMIZATION OF CPG PARAMETERS (8D)")
    print("=" * 70)
    print(f"  Target fwd pos  : {target_forward_position} m  (lateral target = 0)")
    print(f"  Robot mass      : {robot_mass} kg")
    print(f"  Total trials    : {n_trials}")
    print(f"  Optimizer       : {optimizer_name},  Seed: {seed}")
    print(f"  Du              : {Du},  Dy: {Dy}")
    print(f"  Prior mu0       : {mu0}")
    print(f"  Results CSV     : {csv_path}")
    print("=" * 70)

    for trial_idx in range(1, n_trials + 1):
        agent.reset_buffer()

        J, metrics = evaluate_candidate(
            params_np               = None,
            target_forward_position = target_forward_position,
            robot_mass              = robot_mass,
            optimizer_name          = optimizer_name,
            seed                    = seed,
            trial_idx               = trial_idx,
            agent                   = agent,
        )

        csv_writer.writerow(metrics)
        csv_file.flush()

        params8d = np.array([
            metrics["couplinggain"], metrics["wswing"],   metrics["wstance"],
            metrics["FFAST"],        metrics["STOPGAIN"],
            metrics["hipamplitude"], metrics["kneeamplitude"], metrics["b"],
        ])

        x_torch      = torch.tensor(params8d, dtype=dtype)
        train_X_orig = torch.cat([train_X_orig, x_torch.unsqueeze(0)], dim=0)
        train_Y      = torch.cat([train_Y,
                                   torch.tensor([[J]], dtype=dtype)], dim=0)

        fall_status = "FELL" if metrics["fell"] else "OK"
        print(f"\nTrial {trial_idx}/{n_trials}:")
        print("  Params: " + ", ".join(
            f"{n}={v:.3f}" for n, v in zip(param_names, params8d)))
        print(f"  J={J:.4f} [{fall_status}]  "
              f"Dist={metrics['forwarddistance']:.3f}m  "
              f"Vel={metrics['meanvx']:.3f}m/s  "
              f"CoT={metrics['CoT']:.2f}  "
              f"Stab={metrics['stabilityindex']:.2f}°")
        if trial_idx > 1:
            best_J   = train_Y.max().item()
            best_idx = train_Y.argmax().item()
            print(f"  Best so far: J={best_J:.4f} (trial {best_idx+1})")

    csv_file.close()
    print(f"\n✅ CSV saved: {csv_path}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    best_J      = train_Y.max().item()
    best_idx    = train_Y.argmax().item()
    best_params = train_X_orig[best_idx].numpy()
    print(f"Best J = {best_J:.4f}  (trial {best_idx+1})")
    print("Best parameters:")
    for name, value in zip(param_names, best_params):
        print(f"  {name:16s} = {value:.4f}")
    print("=" * 70)

    return train_X_orig, train_Y, best_params


# =============================================================================
# PLOTTING — mirrors BO Figures 1 / 2 / 3
# =============================================================================

def plot_marxefe_results(csv_path: str,
                         target_forward_position: float = 4.0,
                         save_prefix: str = "marxefe") -> None:
    """Produce the same three figures as the BO pipeline from a MARXEFE CSV."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("⚠️  CSV is empty – nothing to plot.")
        return

    n_trials = len(rows)
    trials   = np.arange(1, n_trials + 1)

    def _f(key):
        return np.array([float(r[key]) for r in rows])

    objectives        = _f("J")
    cots              = _f("CoT")
    forward_distances = _f("forwarddistance")
    lateral_drifts    = _f("lateraldrift")
    mean_velocities   = _f("meanvx")
    stabilities       = _f("stabilityindex")
    fall_flags        = _f("fell")

    param_names  = ["couplinggain", "wswing", "wstance", "FFAST",
                    "STOPGAIN", "hipamplitude", "kneeamplitude", "b"]
    param_arrays = [_f(n) for n in param_names]

    cumulative_falls = np.cumsum(fall_flags)
    fall_rate        = cumulative_falls / trials * 100
    best_J_curve     = np.maximum.accumulate(objectives)
    best_J           = objectives.max()
    best_idx         = int(objectives.argmax())
    best_params      = [arr[best_idx] for arr in param_arrays]

    # Figure 1
    fig1, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig1.suptitle('MARXEFE Performance Metrics Over Optimization',
                  fontsize=16, fontweight='bold')

    ax = axes[0]
    ax.plot(trials, forward_distances, 'o-', lw=2, ms=4, color='steelblue',
            label='Forward Distance (+Y)')
    ax.axhline(forward_distances.max(), color='green', ls='--', lw=1.5,
               label=f'Max: {forward_distances.max():.3f} m', alpha=0.7)
    ax.set(xlabel='Trial Number', ylabel='Forward Distance [m]',
           title='Forward Distance (straight motion, +Y axis)')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(trials, lateral_drifts, 'o-', lw=2, ms=4, color='orange',
            label='Lateral Drift (X)')
    valid_d = lateral_drifts[lateral_drifts < 10]
    if len(valid_d):
        ax.axhline(valid_d.min(), color='green', ls='--', lw=1.5,
                   label=f'Min: {valid_d.min():.3f} m', alpha=0.7)
    ax.set(xlabel='Trial Number', ylabel='Lateral Drift [m]',
           title='Lateral Drift (X-axis deviation)')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(trials, stabilities, 'o-', lw=2, ms=4, color='purple',
            label='Combined RMS Stability')
    valid_s = stabilities[stabilities < 100]
    if len(valid_s):
        ax.axhline(valid_s.min(), color='green', ls='--', lw=1.5,
                   label=f'Min: {valid_s.min():.2f}°', alpha=0.7)
    ax.set(xlabel='Trial Number', ylabel='Combined RMS Stability [deg]',
           title='Stability Metric')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(trials, fall_rate, lw=2.5, color='crimson',
            label='Cumulative Fall Rate')
    ax.fill_between(trials, 0, fall_rate, alpha=0.2, color='crimson')
    ax.set(xlabel='Trial Number', ylabel='Fall Rate [%]',
           title=f'Cumulative Fall Rate (Final: {fall_rate[-1]:.1f}%)')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    out1 = f'{save_prefix}_fig1_performance.png'
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"✅ Saved {out1}")

    # Figure 2
    param_labels = ["Coupling Gain", "ω_swing", "ω_stance", "F_FAST",
                    "STOP_GAIN", "Hip Amplitude", "Knee Amplitude",
                    "b (Sharpness)"]
    fig2, axes2 = plt.subplots(3, 3, figsize=(16, 10))
    fig2.suptitle('MARXEFE CPG Parameter Evolution', fontsize=14, fontweight='bold')

    for i, (lbl, arr) in enumerate(zip(param_labels, param_arrays)):
        ax = axes2.flat[i]
        ax.plot(trials, arr, 'o-', lw=1.5, ms=4, alpha=0.7)
        ax.axhline(best_params[i], color='red', ls='--', lw=2,
                   label=f'Best: {best_params[i]:.3f}', alpha=0.8)
        ax.axhline(bounds[0, i].item(), color='gray', ls=':', lw=1, alpha=0.5)
        ax.axhline(bounds[1, i].item(), color='gray', ls=':', lw=1, alpha=0.5)
        ax.set(xlabel='Trial', ylabel=lbl, title=lbl)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    axes2.flat[-1].set_visible(False)
    plt.tight_layout()
    out2 = f'{save_prefix}_fig2_parameters.png'
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"✅ Saved {out2}")

    # Figure 3
    fig3, axes3 = plt.subplots(4, 1, figsize=(16, 12))
    fig3.suptitle('MARXEFE Optimization Metrics: Objective, CoT, Velocity, Stability',
                  fontsize=16, fontweight='bold')

    ax = axes3[0]
    ax.plot(trials, objectives, 'o-', lw=2, ms=4, color='purple',
            label='J per trial')
    ax.plot(trials, best_J_curve, '-', lw=2, color='green',
            label=f'Best so far (max={best_J:.2f})')
    ax.set(xlabel='Trial Number', ylabel='Objective J',
           title='Objective Function J')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes3[1]
    cots_viz = np.clip(cots, 0, 50)
    ax.plot(trials, cots_viz, 'o-', lw=2, ms=4, color='darkorange', label='CoT')
    valid_c = cots[cots < 50]
    if len(valid_c):
        ax.axhline(valid_c.min(), color='green', ls='--', lw=1.5,
                   label=f'Min: {valid_c.min():.3f}', alpha=0.7)
    ax.set(xlabel='Trial Number', ylabel='Cost of Transport',
           title='Cost of Transport (lower is better)')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes3[2]
    implied_target_velocity = target_forward_position / 4.5
    ax.plot(trials, mean_velocities, 'o-', lw=2, ms=4, color='teal',
            label='Mean Forward Velocity [m/s]')
    ax.axhline(implied_target_velocity, color='red', ls='--', lw=1.5,
               label=f'Implied target: {implied_target_velocity:.2f} m/s '
                     f'(={target_forward_position:.1f} m / 4.5 s)', alpha=0.7)
    ax.set(xlabel='Trial Number', ylabel='Mean Forward Velocity [m/s]',
           title='Mean Forward Velocity (+Y axis)')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes3[3]
    stab_viz = np.clip(stabilities, 0, 20)
    ax.plot(trials, stab_viz, 'o-', lw=2, ms=4, color='magenta',
            label='RMS Stability')
    valid_sv = stabilities[stabilities < 20]
    if len(valid_sv):
        ax.axhline(valid_sv.min(), color='green', ls='--', lw=1.5,
                   label=f'Min: {valid_sv.min():.2f}°', alpha=0.7)
    ax.set(xlabel='Trial Number', ylabel='Combined RMS Stability [deg]',
           title='Combined RMS Stability (Roll + Pitch)')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    out3 = f'{save_prefix}_fig3_metrics.png'
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    print(f"✅ Saved {out3}")
