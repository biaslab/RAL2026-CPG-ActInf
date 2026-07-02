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
import casadi as ca
from numpy.linalg import slogdet
from scipy.linalg import det, inv
from scipy.special import gamma, gammaln
from scipy.stats import multivariate_normal

from methods.cpg_bounds import (bounds, bounds_lower, bounds_upper,
                                ALPHA_HOPF, PHI_TROT, THETA_TROT_INIT,
                                H_LEG, D_STEP, leg_ik)
from methods import terrain


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
                 num_iters=10,
                 forgetting=1.0):

        # Exponential forgetting factor λ ∈ (0, 1] applied to the accumulated
        # sufficient statistics on every update. λ = 1 recovers the standard
        # (non-forgetting) matrix-normal update; λ < 1 makes the posterior decay
        # old data with effective window ~1/(1-λ) samples, so the model can
        # track non-stationary dynamics (e.g. flat → slope) online.
        self.forgetting = float(forgetting)
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
        # Keep the initial prior so forgetting decays the posterior *toward the
        # prior* (not toward zero); this keeps Λ and Ω positive-definite and
        # numerically stable even with poorly-excited regressors.
        self.Λ_prior = np.array(coefficients_row_covariance, dtype=float).copy()
        self.Ω_prior = np.array(precision_scale, dtype=float).copy()
        self.ν_prior = float(precision_degrees)
        self.μ = control_prior_mean
        self.Υ = control_prior_precision
        self.goal_prior = goal_prior
        self.thorizon = time_horizon
        self.num_iters = num_iters
        self.free_energy = float('inf')
        # Count of data points assimilated (persists across trials, unlike the
        # buffers). Used to decide when the model is informed enough to drive
        # action selection — robust to forgetting, which keeps ν bounded.
        self.n_updates = 0
        # Cache of prediction constants (inverses, M.T, ...). The posterior is
        # fixed during a prediction/EFE rollout, so these are computed once and
        # reused; invalidated (set to None) whenever the posterior changes.
        self._const = None
        # Cached parametric EFE solver (built once, reused with new numeric
        # parameters + warm start) and the last solution for warm-starting.
        self._efe_solver = None
        self._efe_sig = None
        self._efe_u_prev = None

    def set_goal_velocity(self, target_velocity):
        """Switch the forward-velocity target (non-stationary task). Replaces the
        goal-prior mean (same covariance) and invalidates the cached EFE solver,
        which bakes m_star in as a constant — so the next minimizeEFE() rebuilds
        with the new target. The learned dynamics model is kept, so the agent
        re-plans for the new speed using everything it already knows."""
        cov = np.asarray(self.goal_prior.cov, float)
        mean = np.asarray(self.goal_prior.mean, float).copy()
        mean[0] = float(target_velocity)
        self.goal_prior = multivariate_normal(mean, cov)
        self._efe_solver = None
        self._efe_sig = None

    def update(self, y_k, u_k):
        λ = self.forgetting
        # Forget toward the prior: blend the current posterior statistics with
        # the prior by (1-λ) before assimilating the new datum. λ = 1 leaves the
        # statistics unchanged (standard non-forgetting update); λ < 1 caps the
        # effective memory at ~1/(1-λ) samples while keeping Λ0, Ω0 ⪰ (1-λ)·prior
        # so both stay positive-definite.
        M0 = self.M
        Λ0 = λ * self.Λ + (1.0 - λ) * self.Λ_prior
        Ω0 = λ * self.Ω + (1.0 - λ) * self.Ω_prior
        ν0 = λ * self.ν + (1.0 - λ) * self.ν_prior

        self.ubuffer = self.backshift(self.ubuffer, u_k)
        x_k = np.concatenate([self.ubuffer.flatten(), self.ybuffer.flatten()])

        X = np.outer(x_k, x_k)
        Ξ = np.outer(x_k, y_k) + np.dot(Λ0, M0)

        Λ_new = Λ0 + X
        self.ν = ν0 + 1
        self.Λ = Λ_new
        Ω_new = (Ω0 + np.outer(y_k, y_k) + np.dot(M0.T, np.dot(Λ0, M0))
                 - np.dot(Ξ.T, np.dot(inv(Λ_new), Ξ)))
        self.Ω = 0.5 * (Ω_new + Ω_new.T)   # symmetrize against round-off drift
        self.M = np.dot(inv(Λ_new), Ξ)

        self.ybuffer = self.backshift(self.ybuffer, y_k)
        self.n_updates += 1
        self._const = None   # posterior changed -> invalidate prediction cache
        return None

    def params(self):
        return self.M, self.U, self.V, self.ν

    def log_evidence(self, y, x):
        η, μ, Ψ = self.posterior_predictive(x)
        return -0.5 * (self.Dy * np.log(η * np.pi) - np.log(det(Ψ)) - 2 * self.logmultigamma(self.Dy, (η + self.Dy) / 2) +
                       2 * self.logmultigamma(self.Dy, (η + self.Dy - 1) / 2) + (η + self.Dy) * np.log(1 + 1 / η * np.dot((y - μ).T, np.dot(Ψ, (y - μ)))))

    def predictive_constants(self):
        """Precompute (and cache) the constants the predictive distribution and
        EFE need. The posterior (M, Λ, Ω, ν) is fixed during a prediction / EFE
        rollout, so these are computed once and reused; the cache is invalidated
        on every `update`. All values are numeric, so they can be handed to
        CasADi as `ca.DM` constants — leaving only matmuls in the symbolic graph.
        """
        if self._const is None:
            inv_O = inv(self.Ω)
            self._const = {
                "M_T":              self.M.T,                       # (Dy, Dx)
                "inv_Lambda":       inv(self.Λ),                    # (Dx, Dx)
                "inv_Omega":        inv_O,                          # (Dy, Dy)
                "Omega":            np.asarray(self.Ω),             # (Dy, Dy)
                "eta":              float(self.ν - self.Dy + 1),
                "logdet_inv_Omega": float(slogdet(inv_O)[1]),
            }
        return self._const

    def posterior_predictive(self, x_t):
        """Student-t predictive: returns (eta, mean, precision Psi).

        Uses cached constants — no matrix inversion per call. With
        s = 1 + xᵀΛ⁻¹x: mean = Mᵀx, Psi = (eta/s)·Ω⁻¹, and the predictive
        covariance is the closed form s/(eta-2)·Ω (see `predictions`).
        """
        c = self.predictive_constants()
        eta = c["eta"]
        x_t = np.asarray(x_t, dtype=float)
        mu_t = c["M_T"] @ x_t
        scale = 1.0 + float(x_t @ (c["inv_Lambda"] @ x_t))
        Psi_t = (eta / scale) * c["inv_Omega"]
        return eta, mu_t, Psi_t

    def predictions(self, controls, time_horizon=None):
        """Roll the posterior predictive forward H steps under `controls`
        (shape (Du, H); (H, Du) or flat length H·Du are also accepted). Returns
        per-step mean m_y (Dy, H) and covariance S_y (Dy, Dy, H).

        Optimized for speed and CasADi readiness:
          * constants (Mᵀ, Λ⁻¹, Ω, eta) are precomputed once via
            `predictive_constants` — no inversion inside the loop;
          * the predictive covariance uses the closed form
            S = [scale/(eta-2)]·Ω, scale = 1 + xᵀΛ⁻¹x (no inverse of Psi);
          * the per-step map is purely mean = Mᵀx and scale = 1 + xᵀΛ⁻¹x, i.e.
            matmuls with constant matrices, so these exact expressions transcribe
            directly into a CasADi graph (constants as ca.DM, controls as ca.MX)
            — the same form already used inside `minimizeEFE`.
        """
        H = self.thorizon if time_horizon is None else int(time_horizon)
        controls = np.atleast_2d(np.asarray(controls, dtype=float))
        if controls.shape != (self.Du, H):       # accept (H, Du) or flat
            controls = controls.reshape(H, self.Du).T

        c = self.predictive_constants()
        eta, M_T, inv_L, Omega = c["eta"], c["M_T"], c["inv_Lambda"], c["Omega"]
        cov_factor = 1.0 / (eta - 2.0)

        m_y = np.zeros((self.Dy, H))
        S_y = np.zeros((self.Dy, self.Dy, H))
        ubuf = self.ubuffer.copy()
        ybuf = self.ybuffer.copy()
        for t in range(H):
            ubuf = self.backshift(ubuf, controls[:, t])
            x_t = np.concatenate([ubuf.flatten(), ybuf.flatten()])
            mu_t = M_T @ x_t
            scale = 1.0 + float(x_t @ (inv_L @ x_t))
            m_y[:, t] = mu_t
            S_y[:, :, t] = (scale * cov_factor) * Omega
            ybuf = self.backshift(ybuf, mu_t)
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

    def _build_efe_solver(self, lambda_energy, max_iter, tol, verbose):
        """Build (once) a parametric IPOPT solver for the EFE problem.

        The posterior constants (Mᵀ, Λ⁻¹, eta, logdet Ω⁻¹, tr[S⁻¹Ω]) and the AR
        buffers are CasADi *parameters* `p`, so the compiled solver is reused
        across timesteps / trials — only the numeric `p` and the warm-start `x0`
        change. The goal and control-prior terms are fixed at construction, so
        they are baked in as constants. Rebuilt only if the signature (horizon,
        dims, lambda_energy, max_iter, tol) changes.
        """
        Du, Dy, thorizon, Dx = self.Du, self.Dy, self.thorizon, self.Dx
        Wu = self.ubuffer.shape[1]      # delay_inp + 1
        Wy = self.ybuffer.shape[1]      # delay_out
        n_u = thorizon * Du

        u = ca.MX.sym('u', n_u)                       # decision variable
        P_MT  = ca.MX.sym('MT', Dy * Dx)              # time-varying parameters
        P_iL  = ca.MX.sym('iL', Dx * Dx)
        P_eta = ca.MX.sym('eta', 1)
        P_ld  = ca.MX.sym('ld', 1)
        P_tr  = ca.MX.sym('tr', 1)
        P_ub  = ca.MX.sym('ub', Du * Wu)
        P_yb  = ca.MX.sym('yb', Dy * Wy)
        p_sym = ca.vertcat(P_MT, P_iL, P_eta, P_ld, P_tr, P_ub, P_yb)

        M_T   = ca.reshape(P_MT, Dy, Dx)
        inv_L = ca.reshape(P_iL, Dx, Dx)
        eta, logdet, trv = P_eta, P_ld, P_tr
        ubuf  = ca.reshape(P_ub, Du, Wu)
        ybuf  = ca.reshape(P_yb, Dy, Wy)

        m_star     = ca.DM(np.asarray(self.goal_prior.mean, float).reshape(-1, 1))
        inv_S_star = ca.DM(np.linalg.inv(self.goal_prior.cov))
        mu_prior   = ca.DM(np.asarray(self.μ, float).reshape(-1, 1))
        Upsilon    = ca.DM(np.asarray(self.Υ, float))

        J = ca.MX(0)
        for t in range(thorizon):
            u_t = u[t * Du:(t + 1) * Du]
            ubuf = ca.horzcat(u_t, ubuf[:, :-1])
            x_t = ca.vertcat(ca.reshape(ubuf.T, -1, 1), ca.reshape(ybuf.T, -1, 1))
            scale = 1.0 + (x_t.T @ inv_L @ x_t)
            mi = Dy * ca.log(eta) - Dy * ca.log(scale) + logdet
            mu_t = M_T @ x_t
            diff = mu_t - m_star
            ce = 0.5 * (scale / (eta - 2) * trv + diff.T @ inv_S_star @ diff)
            up_diff = u_t - mu_prior
            cp = 0.5 * (up_diff.T @ Upsilon @ up_diff)
            J = J + mi + ce + cp
            if lambda_energy != 0.0:
                J = J + lambda_energy * (u_t.T @ u_t)
            ybuf = ca.horzcat(mu_t, ybuf[:, :-1])

        opts = {
            'print_time': False,
            'ipopt.print_level': 5 if verbose else 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': int(max_iter),
            'ipopt.tol': float(tol),
            'ipopt.acceptable_tol': max(10.0 * float(tol), 1e-2),
            'ipopt.acceptable_iter': 5,
            # exact Hessian (auto-diff): the control vector is small (thorizon*Du),
            # so exact 2nd-order converges in far fewer iterations than L-BFGS.
        }
        self._efe_solver = ca.nlpsol(
            'efe_solver', 'ipopt', {'x': u, 'f': J, 'p': p_sym}, opts)
        self._efe_sig = (thorizon, Du, Dy, Dx, Wu, Wy,
                         float(lambda_energy), int(max_iter), float(tol), bool(verbose))

    def minimizeEFE(self, u_0=None, verbose=False, control_lims=(-np.inf, np.inf),
                    lambda_energy=0.0, max_iter=100, tol=1e-3, warm_start=True):
        """Minimize EFE via a cached, warm-started parametric IPOPT solver.

        The compiled solver is built once per (horizon, dims, lambda_energy,
        max_iter, tol) and reused; each call only updates the numeric parameters
        (posterior constants + AR buffers) and the warm-start point. Warm start
        is, in priority: explicit `u_0` → the previous solution → the control
        prior mean. Returns the flat optimal control sequence (length
        thorizon*Du). `control_lims` is a single (lo, hi) tuple or a per-dim list.
        """
        Du, thorizon = self.Du, self.thorizon
        n_u = thorizon * Du

        sig = (thorizon, Du, self.Dy, self.Dx, self.ubuffer.shape[1],
               self.ybuffer.shape[1], float(lambda_energy), int(max_iter),
               float(tol), bool(verbose))
        if self._efe_solver is None or self._efe_sig != sig:
            self._build_efe_solver(lambda_energy, max_iter, tol, verbose)

        if (isinstance(control_lims, tuple) and len(control_lims) == 2
                and np.isscalar(control_lims[0])):
            lbx = np.full(n_u, float(control_lims[0]))
            ubx = np.full(n_u, float(control_lims[1]))
        else:
            lbx = np.array([b[0] for b in control_lims], dtype=float)
            ubx = np.array([b[1] for b in control_lims], dtype=float)

        if u_0 is not None:
            x0 = np.asarray(u_0, dtype=float).reshape(-1)
            if x0.size != n_u:
                x0 = np.tile(np.asarray(self.μ, float), thorizon)
        elif warm_start and self._efe_u_prev is not None and self._efe_u_prev.size == n_u:
            x0 = self._efe_u_prev
        else:
            x0 = np.tile(np.asarray(self.μ, float), thorizon)

        # Pack the time-varying parameters (column-major, to match ca.reshape).
        c = self.predictive_constants()
        inv_S_star = np.linalg.inv(self.goal_prior.cov)
        tr_iSOmega = float(np.trace(inv_S_star @ self.Ω))
        p_val = np.concatenate([
            np.asarray(c["M_T"], float).flatten('F'),
            np.asarray(c["inv_Lambda"], float).flatten('F'),
            [c["eta"]], [c["logdet_inv_Omega"]], [tr_iSOmega],
            np.asarray(self.ubuffer, float).flatten('F'),
            np.asarray(self.ybuffer, float).flatten('F'),
        ])

        sol = self._efe_solver(x0=x0, lbx=lbx, ubx=ubx, p=p_val)
        u_opt = np.array(sol['x']).reshape(-1)
        if warm_start:
            self._efe_u_prev = u_opt
        return u_opt

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


# Control prior hyperparameter: bounds [lower_i, upper_i] correspond to
# mu_i ± n_sigma * sigma_i.
n_sigma = 2.0

quadruped          = None
joint_IDs_full     = None
filtered_joint_IDs = None
feet_joint_IDs     = None

# Parameters held by the previous trial, used to smoothly interpolate the new
# per-trial parameter selection over the transition window (same scheme as the
# BO baseline in run_cpg_trial). Reset to None at the start of each optimization.
_prev_params_marx  = None

# Trained agent from the most recent marxefe_optimize_cpg call (for post-hoc
# evaluation episodes such as transition-recovery traces).
_last_agent        = None


# =============================================================================
# ENVIRONMENT AND ROBOT SETUP
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
    terrain.build_ground(p)   # body 0; flat (plane.urdf) by default
    return p


def load_robot(p, robot_mass=10.0):
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
    p.changeDynamics(robot, -1, mass=float(robot_mass))

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
# STATE OBSERVATION / FALL DETECTION
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
    """Observation y_k = [vx, vy, pitch, roll] for MARXEFE (Dy=4).

    The agent tracks a *target forward velocity*: vx = forward (+Y) base
    velocity, vy = lateral (X) base velocity. Velocity is a stationary,
    well-posed target for the linear AR model — unlike absolute position,
    which is an unbounded integrator state and led the EFE selection to
    extrapolate to destabilising parameters. Absolute base_pos is still
    returned for metric logging and fall detection (the evaluation objective
    J remains position-based and identical across all methods).
    """
    base_pos, base_orientation = get_base_orientation(p, robot, ori_default)
    base_vel, _ = p.getBaseVelocity(robot)
    vx = base_vel[1]   # forward +Y
    vy = base_vel[0]   # lateral  X
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation)
    y_k = np.array([vx, vy, pitch, roll])
    return y_k, base_pos, base_orientation


class JointCPG:
    """Self-contained joint-space Righetti-style CPG (the validated controller).

    Encapsulates the four coupled Hopf oscillators, the Righetti STOP/FAST
    contact feedback, the per-leg phase hysteresis, and the CPG→joint mapping.
    One instance per episode; call :meth:`step` each control tick.

    8-D parameter vector (matches ``methods.cpg_bounds``):
      [coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN, hip_amp, knee_amp, b]

    This is the joint-space mapping shared with ``methods.bo_optimizer`` — it
    replaces the Zhang-et-al. Cartesian foot-trajectory + IK controller, which
    was laterally unstable on Laikago under position control.
    """

    ALPHA = 3.0
    BETA = 12.0
    U = 2.0
    HIP_OFFSET = 0.26
    KNEE_OFFSET = -1.0
    SWING_ENTER, SWING_EXIT = 0.15, 0.02
    STANCE_ENTER, STANCE_EXIT = -0.15, -0.02
    DEBOUNCE_THRESHOLD = 2
    K = np.array([[0, -1, -1, 1],
                  [-1, 0, 1, -1],
                  [-1, 1, 0, -1],
                  [1, -1, -1, 0]], dtype=float)

    def __init__(self, n_legs=4):
        self.n = n_legs
        theta = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        self.x = np.array([np.sqrt(self.U) * np.cos(t) for t in theta])
        self.y = np.array([np.sqrt(self.U) * np.sin(t) for t in theta])
        self.deb = np.zeros(n_legs, dtype=int)
        self.cc = np.zeros(n_legs, dtype=int)
        self.phase = []
        for j in range(n_legs):
            if self.y[j] > self.SWING_ENTER:
                self.phase.append("swing")
            elif self.y[j] < self.STANCE_ENTER:
                self.phase.append("stance")
            else:
                self.phase.append("transition")

    def _get_phase(self, y_val, j):
        s = self.phase[j]
        if s == "swing":
            s = "transition" if y_val < self.SWING_EXIT else "swing"
        elif s == "stance":
            s = "transition" if y_val > self.STANCE_EXIT else "stance"
        else:
            if y_val > self.SWING_ENTER:
                s = "swing"
            elif y_val < self.STANCE_ENTER:
                s = "stance"
            else:
                s = "transition"
        self.phase[j] = s
        return s

    def step(self, params_8d, raw_contacts, dt):
        """Advance one control tick; return (hip_angles, knee_angles) arrays."""
        coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN, hip_amp, knee_amp, b = params_8d
        x_prev, y_prev = self.x.copy(), self.y.copy()

        # radial state + intrinsic frequency (swing/stance blended by y)
        w_vec = w_stance / (np.exp(-b * y_prev) + 1.0) + w_swing / (np.exp(b * y_prev) + 1.0)
        r_vec = np.sqrt(x_prev ** 2 + y_prev ** 2)
        x_new = x_prev + dt * (self.ALPHA * (self.U - r_vec ** 2) * x_prev - w_vec * y_prev)

        # contact debounce
        raw = np.asarray(raw_contacts, dtype=int)
        for j in range(self.n):
            if raw[j] == self.deb[j]:
                self.cc[j] = 0
            else:
                self.cc[j] += 1
            if self.cc[j] >= self.DEBOUNCE_THRESHOLD:
                self.deb[j] = raw[j]
                self.cc[j] = 0

        # STOP/FAST feedback per leg
        phases = [self._get_phase(y_prev[j], j) for j in range(self.n)]
        coupling_y = coupling_gain * (self.K @ y_prev)
        u_fb = np.zeros(self.n)
        for j in range(self.n):
            in_stop = ((phases[j] == "swing" and self.deb[j] < 0.5) or
                       (phases[j] == "stance" and self.deb[j] > 0.5))
            if in_stop:
                u_fb[j] = STOP_GAIN * (w_vec[j] * x_prev[j] - coupling_y[j])
            elif phases[j] in ("swing", "stance"):
                u_fb[j] = np.sign(y_prev[j]) * F_FAST

        y_new = y_prev + dt * (self.BETA * (self.U - r_vec ** 2) * y_prev
                               + w_vec * x_prev + coupling_y + u_fb)

        self.x, self.y = x_new, y_new
        hip_angles = self.HIP_OFFSET + hip_amp * x_new
        knee_angles = self.KNEE_OFFSET - knee_amp * np.maximum(0.0, y_new)
        return hip_angles, knee_angles


def reset_simulation(p, robot, filtered_joint_IDs, ori_default):
    """Reset robot to neutral standing pose, settle, and return a fresh JointCPG."""
    start_position = [0.0, 0.0, 0.55]
    p.resetBasePositionAndOrientation(robot, start_position, ori_default)
    p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])

    abduction_joint_ids = [0, 4, 8, 12]
    hip_joint_ids       = [1, 5, 9, 13]
    knee_joint_ids      = [2, 6, 10, 14]

    # Joint-space CPG stance pose (matches the BO pipeline's validated reset).
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

    return JointCPG(n_legs=4)


# =============================================================================
# MARXEFE EPISODE
# =============================================================================

def run_episode_maxrefe(agent, robot, joint_IDs_full_arg,
                        filtered_joint_IDs_arg, feet_joint_IDs_arg,
                        dt, episode_length=4.5,
                        lambda_energy=1e-2,
                        target_forward_position=4.0,
                        update_every=0, ramp_steps=20,
                        target_schedule=None,
                        debug=False):
    """Run one MARXEFE-controlled episode.
    Trial structure: Phase1(100 steps settle, in reset) + Phase2(150 transition)
    + Phase3(300 steady) = 4.5 s at dt=0.01.

    update_every : int
        0  -> select parameters once per trial and hold them (BO-like; the
              flat-terrain protocol).
        >0 -> re-select via EFE every `update_every` steps for *online*
              within-episode adaptation (e.g. mid-episode terrain changes),
              ramping each new selection in over `ramp_steps` steps.
    """
    ori_default = [0.0, 0.5, 0.5, 0.0]

    cpg = reset_simulation(p, robot, filtered_joint_IDs_arg, ori_default)

    n_legs = 4
    y_k, base_pos, base_orientation = extract_observation(p, robot, ori_default)

    transition_duration = 1.5
    trial_duration      = episode_length
    num_steps           = int(trial_duration / dt)
    transition_steps    = int(transition_duration / dt)

    # ------------------------------------------------------------------
    # ACTION SELECTION (active inference)
    # The 8 CPG parameters are gait hyperparameters, selected by minimising
    # expected free energy under the current posterior. With update_every == 0
    # they are chosen ONCE per trial and held, interpolated from the previous
    # trial over the transition window (BO-like). With update_every > 0 they
    # are RE-selected every `update_every` steps so the agent adapts online
    # within the episode; each new selection is ramped in over `ramp_steps`
    # steps to avoid the bang-bang chatter that toppled the robot when
    # re-optimising every step. The posterior is updated every step regardless,
    # so a mid-episode terrain change is reflected in the next selection.
    # ------------------------------------------------------------------
    global _prev_params_marx
    bounds_agent = [(bounds_lower[i].item(), bounds_upper[i].item())
                    for _ in range(agent.thorizon) for i in range(8)]
    warmup_updates = 10   # need a little data before the model can drive control

    def select_params():
        """Minimise EFE under the current posterior; fall back to the control
        prior mean if the model is still uninformative or the solve fails."""
        if agent.n_updates < warmup_updates:
            return np.array(agent.μ, dtype=float)
        try:
            # No explicit u_0 -> warm-start from the previous solution; loosened
            # tolerance + cached parametric solver for a fast receding-horizon solve.
            u_opt = agent.minimizeEFE(
                control_lims  = bounds_agent,
                lambda_energy = lambda_energy,
                max_iter      = 100,
                tol           = 1e-3,
            )
            return np.clip(np.asarray(u_opt[:8], dtype=float),
                           bounds_lower.numpy(), bounds_upper.numpy())
        except Exception as e:
            print(f"Warning: EFE selection failed: {e}; using control prior mean.")
            return np.array(agent.μ, dtype=float)

    # Initial (trial-start) selection, ramped in from the previous trial's
    # parameters over the long transition window.
    seg_target = select_params()
    seg_start  = (np.array(_prev_params_marx, dtype=float)
                  if _prev_params_marx is not None else seg_target.copy())
    seg_anchor = 0
    seg_ramp   = transition_steps
    applied    = seg_start.copy()

    times       = np.zeros(num_steps)
    y_history   = np.zeros((4, num_steps))
    u_history   = np.zeros((8, num_steps))
    preds_m     = np.zeros((4, num_steps))
    preds_v     = np.zeros((4, num_steps))
    goals_m     = np.zeros((4, num_steps))
    positions   = []

    roll_angles  = np.zeros(num_steps)
    pitch_angles = np.zeros(num_steps)
    yaw_angles   = np.zeros(num_steps)
    target_log   = np.zeros(num_steps)
    tsched = None if target_schedule is None else np.asarray(target_schedule, float)

    if debug:
        post_M_norm   = np.zeros(num_steps)
        post_L_logdet = np.zeros(num_steps)
        post_O_tr     = np.zeros(num_steps)
        post_O_logdet = np.zeros(num_steps)
        post_nu       = np.zeros(num_steps)

    n_joints    = 2 * n_legs
    torques_log = np.zeros((n_joints, num_steps))
    qdot_log    = np.zeros((n_joints, num_steps))
    base_pos_log = np.zeros((num_steps, 3))
    vx_log      = np.zeros(num_steps)
    vy_log      = np.zeros(num_steps)

    is_fallen = False

    leg_names      = ["FL", "FR", "RL", "RR"]
    hip_joint_ids  = [1, 5, 9, 13]
    knee_joint_ids = [2, 6, 10, 14]

    DEBOUNCE_THRESHOLD   = 2
    debounced_contacts   = np.zeros(n_legs, dtype=int)
    contact_change_count = np.zeros(n_legs, dtype=int)
    for k_step in range(num_steps):
        t = k_step * dt
        # Non-stationary task: switch the velocity target when the schedule
        # changes (rebuilds the EFE solver with the new goal; model is kept).
        if tsched is not None:
            v_now = float(tsched[min(k_step, len(tsched) - 1)])
            if abs(v_now - float(agent.goal_prior.mean[0])) > 1e-9:
                agent.set_goal_velocity(v_now)
        target_log[k_step] = float(agent.goal_prior.mean[0])
        times[k_step]      = t
        y_history[:, k_step] = y_k
        goals_m[:, k_step]   = agent.goal_prior.mean

        x_k              = np.concatenate([agent.ubuffer.flatten(),
                                            agent.ybuffer.flatten()])
        eta_k, mu_k, Psi_k = agent.posterior_predictive(x_k)
        preds_m[:, k_step]  = mu_k
        preds_v[:, k_step]  = np.diag(inv(Psi_k) * eta_k / (eta_k - 2))

        # Re-select parameters periodically for online within-episode adaptation
        # (update_every > 0); otherwise the initial trial selection is held.
        if update_every and k_step > 0 and (k_step % update_every == 0):
            seg_start  = applied.copy()
            seg_target = select_params()
            seg_anchor = k_step
            seg_ramp   = max(1, ramp_steps)

        frac = min(1.0, (k_step - seg_anchor) / max(1, seg_ramp))
        applied = seg_start + frac * (seg_target - seg_start)
        applied = np.clip(applied, bounds_lower.numpy(), bounds_upper.numpy())
        params_8d = applied

        # Raw foot contacts for the CPG's STOP/FAST feedback
        raw_contacts = np.array([
            int(len(p.getContactPoints(bodyA=0, bodyB=robot,
                                       linkIndexA=-1, linkIndexB=feet_joint_IDs_arg[j])) > 0)
            for j in range(n_legs)
        ])

        # Joint-space CPG step → hip/knee targets
        hip_angles, knee_angles = cpg.step(params_8d, raw_contacts, dt)
        for j in range(n_legs):
            abd_joint, hip_joint, kn_joint = joint_IDs_full_arg[leg_names[j]]
            p.setJointMotorControl2(robot, abd_joint, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, hip_joint, p.POSITION_CONTROL, hip_angles[j])
            p.setJointMotorControl2(robot, kn_joint,  p.POSITION_CONTROL, knee_angles[j])

        # Spatially-varying friction: set the ground friction from the robot's
        # current forward position (no-op unless this is a friction terrain).
        terrain.apply_dynamic_friction(p, robot, base_pos[1])
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

        roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation)
        roll_angles[k_step]  = roll
        pitch_angles[k_step] = pitch
        yaw_angles[k_step]   = yaw

        agent.update(y_k_new, params_8d)
        u_history[:, k_step] = params_8d

        if debug:
            post_M_norm[k_step] = float(np.linalg.norm(agent.M, 'fro'))
            sgnL, ldL = slogdet(agent.Λ)
            post_L_logdet[k_step] = ldL if sgnL > 0 else np.nan
            post_O_tr[k_step]     = float(np.trace(agent.Ω))
            sgnO, ldO = slogdet(agent.Ω)
            post_O_logdet[k_step] = ldO if sgnO > 0 else np.nan
            post_nu[k_step]       = float(agent.ν)

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
            yaw_angles    = yaw_angles[:actual_steps]
            target_log    = target_log[:actual_steps]
            vx_log        = vx_log[:actual_steps]
            vy_log        = vy_log[:actual_steps]
            if debug:
                y_history     = y_history[:, :actual_steps]
                u_history     = u_history[:, :actual_steps]
                preds_m       = preds_m[:, :actual_steps]
                preds_v       = preds_v[:, :actual_steps]
                goals_m       = goals_m[:, :actual_steps]
                post_M_norm   = post_M_norm[:actual_steps]
                post_L_logdet = post_L_logdet[:actual_steps]
                post_O_tr     = post_O_tr[:actual_steps]
                post_O_logdet = post_O_logdet[:actual_steps]
                post_nu       = post_nu[:actual_steps]
            num_steps     = actual_steps
            break

    # Carry the final applied parameters into the next trial's ramp start.
    _prev_params_marx = np.array(applied, dtype=float).copy()

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

    if len(base_pos_log) > start_idx:
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
    elif len(base_pos_log) > 0:
        forward_distance = base_pos_log[-1, 1] - base_pos_log[0, 1]
        lateral_drift    = abs(base_pos_log[-1, 0] - base_pos_log[0, 0])
        mean_vx          = 0.0
        CoT              = 200.0
    else:
        forward_distance = 0.0
        lateral_drift    = 0.0
        mean_vx          = 0.0
        CoT              = 1000.0

    trial_data = {
        "t":                   times,
        "pos_x":               base_pos_log[:, 1],   # forward +Y
        "pos_y":               base_pos_log[:, 0],   # lateral  X
        "vx":                  vx_log,
        "vy":                  vy_log,
        "roll":                roll_angles,
        "pitch":               pitch_angles,
        "yaw":                 yaw_angles,
        "target":              target_log,
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
        "dt":                  dt,
        "r_cpg":               np.zeros((n_legs, len(times))),
        "theta_cpg":           np.zeros((n_legs, len(times))),
        "final_params_8d":     params_8d,
    }
    if debug:
        trial_data.update({
            "y_history":     y_history,
            "u_history":     u_history,
            "preds_m":       preds_m,
            "preds_v":       preds_v,
            "goals_m":       goals_m,
            "post_M_norm":   post_M_norm,
            "post_L_logdet": post_L_logdet,
            "post_O_tr":     post_O_tr,
            "post_O_logdet": post_O_logdet,
            "post_nu":       post_nu,
        })
    return trial_data


# =============================================================================
# OBJECTIVE FUNCTION — identical to methods.bo_optimizer.compute_objective
# =============================================================================

def compute_objective(trial_data: dict,
                      target_velocity: float,
                      robot_mass: float,
                      g: float = 9.81) -> float:
    """Velocity-tracking objective J — identical to methods.bo_optimizer.

        J = w1 * Σ_steady dt * min(exp(-‖v_xy - v*_xy‖² / 0.05), l_r) - w2 * CoT

    with v*_xy = [target_velocity, 0], w1 = 1, w2 = 0.5, l_r = 0.85 (Zhang et al.
    IROS 2024, eq. 9). Shared with BO/grid so the three methods score identically.
    """
    t                   = trial_data["t"]
    transition_duration = trial_data["transition_duration"]
    steady_idx          = np.searchsorted(t, transition_duration)

    if steady_idx >= len(t) - 5:
        return -50.0

    dt = trial_data.get("dt", t[1] - t[0] if len(t) > 1 else 0.01)
    vx_steady      = np.asarray(trial_data["vx"][steady_idx:])
    vy_steady      = np.asarray(trial_data["vy"][steady_idx:])
    torques_steady = trial_data["torques"][steady_idx:, :]
    qdot_steady    = trial_data["qdot"][steady_idx:, :]
    bpos_steady    = trial_data["base_pos"][steady_idx:, :]

    w1, w2, l_r = 1.0, 0.5, 0.85
    err = ((vx_steady - target_velocity) ** 2 + vy_steady ** 2) / 0.05
    reward = np.minimum(np.exp(-err), l_r)
    velocity_term = w1 * dt * np.sum(reward)

    mech_power  = np.sum(np.abs(torques_steady * qdot_steady)) * dt
    d           = bpos_steady[-1, 1] - bpos_steady[0, 1]
    CoT_cap     = 200.0 if d < 0.5 else (150.0 if d < 1.5 else 100.0)
    CoT         = min(mech_power / (robot_mass * g * max(abs(d), 0.001)), CoT_cap)

    J = velocity_term - w2 * CoT
    return J


# =============================================================================
# EVALUATE_CANDIDATE
# =============================================================================

def evaluate_candidate(params_np, target_velocity, robot_mass,
                       optimizer_name, seed, trial_idx, agent=None,
                       update_every=0, ramp_steps=20,
                       debug=False, debug_save_prefix=None):
    """Evaluate one MARXEFE episode and return (J, metrics_dict).
    params_np is unused (agent drives action selection internally).
    ``target_velocity`` is the forward speed set-point v*_x [m/s] tracked by the
    objective (and matched to the agent's goal prior).
    If debug=True, posterior/predictive traces are recorded and — when
    debug_save_prefix is set — a debug figure is written.
    """
    sim_start = time.time()
    if agent is None:
        raise NotImplementedError("evaluate_candidate requires an agent.")

    trial_data = run_episode_maxrefe(
        agent, quadruped, joint_IDs_full, filtered_joint_IDs, feet_joint_IDs,
        dt=0.01, episode_length=4.5,
        lambda_energy=1e-2,
        update_every=update_every, ramp_steps=ramp_steps,
        debug=debug,
    )
    sim_time_sec = time.time() - sim_start

    if debug and debug_save_prefix is not None:
        plot_marxefe_debug(trial_data, debug_save_prefix)

    J = compute_objective(trial_data, target_velocity, robot_mass)

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
        "gc":           params8d[0],
        "gp":           params8d[1],
        "omegaswing":   params8d[2],
        "omegastance":  params8d[3],
        "mu":           params8d[4],
        "xofffront":    params8d[5],
        "xoffhind":     params8d[6],
        "sigmaN":       params8d[7],
    }
    return J, metrics


# =============================================================================
# ONLINE / CONTINUOUS-RUN HELPERS
# Shared agent factory + a single long continuous run for the BO-vs-MARXEFE
# adaptation comparison (no per-update resets; methods update at their own
# cadence — BO every 4.5 s, MARX-EFE every 1 s — as in Zhang et al. 2024).
# =============================================================================

def build_marx_agent(target_velocity=1.0, control_prior_scale=0.15,
                     goal_prior_std=(np.sqrt(0.5), np.sqrt(0.5),
                                     np.deg2rad(45), np.deg2rad(45)),
                     input_buffer=3, output_buffer=10, time_horizon=2,
                     nu0=20.0, omega0_scale=1.0, lambda0_scale=1e-3,
                     forgetting=1.0):
    """Construct a MARX-EFE agent (Du=8, Dy=4) with the given buffers / horizon /
    priors. Shared by the per-trial optimizer and the continuous online runner."""
    Mu, My, Dy, Du = int(input_buffer) - 1, int(output_buffer), 4, 8
    reg_dim = Dy * My + Du * (Mu + 1)
    Omega0  = omega0_scale * np.diag(np.ones(Dy))
    Lambda0 = lambda0_scale * np.diag(np.ones(reg_dim))
    Mean0   = 1e-8 * rnd.randn(reg_dim, Dy)
    mu_t    = 0.5 * (bounds_lower + bounds_upper)
    sigma_t = control_prior_scale * (bounds_upper - bounds_lower) / (2.0 * n_sigma)
    Upsilon0 = torch.diag(1.0 / (sigma_t ** 2)).numpy()
    svx, svy, sp, sr = goal_prior_std
    goal = multivariate_normal(np.array([target_velocity, 0.0, 0.0, 0.0]),
                               np.diag([svx**2, svy**2, sp**2, sr**2]))
    return MARXAgent(Mean0.copy(), Lambda0.copy(), Omega0.copy(), float(nu0),
                     mu_t.numpy().copy(), Upsilon0.copy(), goal,
                     Dy=Dy, Du=Du, delay_inp=Mu, delay_out=My,
                     time_horizon=int(time_horizon), forgetting=forgetting)


def _velocity_window_J(vx, vy, torques, qdot, fwd_dist, robot_mass,
                       target_velocity, dt, g=9.81):
    """Scalar windowed objective (velocity tracking minus cost-of-transport)
    that online BO maximizes over each 4.5 s window."""
    T = len(vx)
    if T < 5:
        return -50.0
    err = ((np.asarray(vx) - target_velocity) ** 2 + np.asarray(vy) ** 2) / 0.05
    R_v = float(np.mean(np.minimum(np.exp(-err), 0.85)))
    mech = float(np.sum(np.abs(np.asarray(torques) * np.asarray(qdot))) * dt)
    d = max(float(fwd_dist), 1e-3)
    CoT = min(mech / (robot_mass * g * d), 200.0)
    return 5.0 * R_v - 0.4 * (CoT / 50.0)


def run_bo_online(robot, joint_IDs_full_arg, filtered_joint_IDs_arg,
                  feet_joint_IDs_arg, dt, run_length, bo, target_velocity,
                  robot_mass, window_steps=450, transition_steps=150, seed=0,
                  bo_trust_radius=None, init_params=None,
                  target_schedule=None, param_schedule=None,
                  bo_factory=None, restart_steps=None):
    """One continuous online run (no resets after the start). Every
    `window_steps` (= 4.5 s) BO scores the just-finished window via velocity
    tracking, appends (params, J) to its GP, and proposes the next CPG
    parameters, smoothly interpolated over `transition_steps`. Uses the same CPG
    dynamics/feedback as the MARX episode. Returns per-step logs.

    `init_params` sets the starting (and, with `window_steps` larger than the
    run, the held) parameters — set `window_steps` huge and `bo=None` to run a
    FIXED-parameter bout (no adaptation).

    Non-stationary (velocity-switch) extras:
      * `target_schedule` : per-step array of target forward velocities; used by
        the windowed objective so each window is scored against the current
        target. Falls back to scalar `target_velocity` if None.
      * `param_schedule`  : list of (step, params) for the non-adaptive methods —
        at each listed step the gait is switched (ramped) to `params`. Used to
        feed pre-computed per-segment grid optima ("redo grid after each switch")
        or a single fixed gait. Ignored when `bo` is not None.
      * `bo_factory`/`restart_steps` : for online BO, at each step in
        `restart_steps` the GP is rebuilt fresh (bo = bo_factory()) so BO
        re-optimizes from scratch for the new target — the BO analogue of
        re-running grid search after a velocity switch.
    Returns per-step logs."""
    ori_default = [0.0, 0.5, 0.5, 0.0]
    cpg = reset_simulation(p, robot, filtered_joint_IDs_arg, ori_default)
    n_legs = 4
    y_k, base_pos, base_orientation = extract_observation(p, robot, ori_default)

    leg_names = ["FL", "FR", "RL", "RR"]
    hip_joint_ids = [1, 5, 9, 13]; knee_joint_ids = [2, 6, 10, 14]

    num_steps = int(run_length / dt)
    lower, upper = bounds_lower.numpy(), bounds_upper.numpy()
    mid = 0.5 * (lower + upper)            # sane starting gait (control-prior mean)
    rng = np.random.default_rng(seed)
    n_joints = 2 * n_legs

    t_log = np.zeros(num_steps); vx_log = np.zeros(num_steps); vy_log = np.zeros(num_steps)
    fy_log = np.zeros(num_steps); roll_log = np.zeros(num_steps); pitch_log = np.zeros(num_steps)
    yaw_log = np.zeros(num_steps); tgt_log = np.zeros(num_steps)
    torq = np.zeros((num_steps, n_joints)); qd = np.zeros((num_steps, n_joints))
    params_log = np.zeros((num_steps, 8))

    # Start at a sane walking gait (midpoint) rather than a random draw — in a
    # no-reset continuous run a fatal random init would end the bout immediately.
    # `init_params` overrides the start (e.g. a grid-search winner, or a fixed gait).
    seg_target = (np.asarray(init_params, dtype=float).copy()
                  if init_params is not None else mid.copy())
    seg_start = seg_target.copy(); seg_anchor = 0; seg_ramp = transition_steps
    applied = seg_target.copy()
    win_start = 0; n_obs = 0

    tsched = None if target_schedule is None else np.asarray(target_schedule, float)
    def tgt(i):
        return float(tsched[min(i, len(tsched) - 1)]) if tsched is not None else target_velocity
    psched = dict((int(s), np.asarray(pp, float)) for s, pp in (param_schedule or []))
    rsteps = set(int(s) for s in (restart_steps or []))

    DEBOUNCE_THRESHOLD = 2
    debounced = np.zeros(n_legs, dtype=int); cc = np.zeros(n_legs, dtype=int)
    is_fallen = False; k = 0
    for k in range(num_steps):
        t_log[k] = k * dt
        # Non-adaptive methods: switch to the scheduled (e.g. per-segment grid)
        # gait at the listed steps, ramped over the transition window.
        if k in psched:
            seg_start = applied.copy(); seg_target = np.clip(psched[k], lower, upper)
            seg_anchor = k; seg_ramp = transition_steps
        # Online BO: rebuild the GP from scratch at a velocity switch so it
        # re-optimizes for the new target (the BO analogue of re-running grid).
        if k in rsteps and bo_factory is not None:
            bo = bo_factory(); n_obs = 0; win_start = k
        frac = min(1.0, (k - seg_anchor) / max(1, seg_ramp))
        applied = np.clip(seg_start + frac * (seg_target - seg_start), lower, upper)
        params_log[k] = applied; tgt_log[k] = tgt(k)
        params_8d = applied

        # Raw foot contacts for the CPG's STOP/FAST feedback
        raw_contacts = np.array([
            int(len(p.getContactPoints(bodyA=0, bodyB=robot,
                                       linkIndexA=-1, linkIndexB=feet_joint_IDs_arg[j])) > 0)
            for j in range(n_legs)
        ])

        # Joint-space CPG step → hip/knee targets
        hip_angles, knee_angles = cpg.step(params_8d, raw_contacts, dt)
        for j in range(n_legs):
            ab, hp, kn = joint_IDs_full_arg[leg_names[j]]
            p.setJointMotorControl2(robot, ab, p.POSITION_CONTROL, targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, hp, p.POSITION_CONTROL, hip_angles[j])
            p.setJointMotorControl2(robot, kn, p.POSITION_CONTROL, knee_angles[j])

        terrain.apply_dynamic_friction(p, robot, base_pos[1])
        p.stepSimulation()

        idx = 0
        for jh, jk in zip(hip_joint_ids, knee_joint_ids):
            hs = p.getJointState(robot, jh); ks = p.getJointState(robot, jk)
            torq[k, idx] = hs[3]; torq[k, idx + 1] = ks[3]
            qd[k, idx] = hs[1]; qd[k, idx + 1] = ks[1]; idx += 2

        y_k, base_pos, base_orientation = extract_observation(p, robot, ori_default)
        vx_log[k] = y_k[0]; vy_log[k] = y_k[1]; pitch_log[k] = y_k[2]; roll_log[k] = y_k[3]
        yaw_log[k] = p.getEulerFromQuaternion(base_orientation)[2]
        fy_log[k] = base_pos[1]

        is_fallen, _, _, _ = check_if_fallen(p, robot, base_pos, base_orientation, 0.3, 0.25)
        if is_fallen:
            break

        if k > 0 and (k % window_steps == 0):
            s0, s1 = win_start + transition_steps, k
            if s1 > s0 + 5:
                J = _velocity_window_J(vx_log[s0:s1], vy_log[s0:s1], torq[s0:s1],
                                       qd[s0:s1], fy_log[s1 - 1] - fy_log[s0],
                                       robot_mass, tgt(s1 - 1), dt)
            else:
                J = -50.0
            bo._append(seg_target, J); n_obs += 1
            if n_obs < bo.n_init:
                # mild exploration around the sane gait (avoid fatal random probes)
                nxt = mid + rng.uniform(-0.15, 0.15) * (upper - lower)
            else:
                model = bo.fit_model()
                if bo_trust_radius is None:
                    nxt = bo.from_unit(bo.suggest(model, bo.beta_schedule(n_obs)))
                else:
                    # TuRBO-style trust region: optimize the acquisition only in a
                    # box of half-width `bo_trust_radius` (in unit space) around the
                    # incumbent best, so each update can't jump far -> more caution.
                    xb = bo.train_X_unit[int(bo.train_Y.argmax())].numpy()
                    lo = np.clip(xb - bo_trust_radius, 0.0, 1.0)
                    hi = np.clip(xb + bo_trust_radius, 0.0, 1.0)
                    saved = bo.unit_bounds
                    bo.unit_bounds = torch.tensor([lo, hi], dtype=torch.double)
                    try:
                        nxt = bo.from_unit(bo.suggest(model, bo.beta_schedule(n_obs)))
                    finally:
                        bo.unit_bounds = saved
            seg_start = applied.copy(); seg_target = np.clip(nxt, lower, upper)
            seg_anchor = k; seg_ramp = transition_steps; win_start = k

    L = (k + 1) if is_fallen else num_steps
    return {"t": t_log[:L], "vx": vx_log[:L], "vy": vy_log[:L], "fy": fy_log[:L],
            "roll": roll_log[:L], "pitch": pitch_log[:L], "yaw": yaw_log[:L],
            "params": params_log[:L], "target": tgt_log[:L], "fell": is_fallen}


# =============================================================================
# MARXEFE OPTIMISATION LOOP
# =============================================================================

def marxefe_optimize_cpg(bounds: torch.Tensor,
                         target_forward_position: float,
                         robot_mass: float,
                         debug_first_trial: bool = True,
                         n_trials: int = 200,
                         optimizer_name: str = "MARXEFE",
                         seed: int = 0,
                         results_dir: str = "results",
                         goal_prior_std=(0.5, 0.5, np.deg2rad(10), np.deg2rad(10)),
                         control_prior_scale: float = 1.0,
                         target_velocity: float = 1.0,
                         update_every: int = 0,
                         ramp_steps: int = 20,
                         forgetting: float = 1.0,
                         time_horizon: int = 2,
                         input_buffer: int = 3,
                         output_buffer: int = 10,
                         nu0: float = 20.0,
                         omega0_scale: float = 1.0,
                         lambda0_scale: float = 1e-3,
                         ) -> tuple:
    """MARXEFE optimisation loop with BO-compatible CSV logging.

    The agent's internal goal is a target *forward velocity* (m/s); the
    evaluation objective J stays position-based (target_forward_position) so
    all three methods are scored identically.
    """
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
        "gc", "gp", "omegaswing",
        "omegastance", "mu",
        "xofffront", "xoffhind", "sigmaN",
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
         feet_joint_IDs) = load_robot(p, robot_mass=robot_mass)
        print(f"\n✅ Environment initialized with robot ID: {quadruped} (mass={robot_mass} kg)")

    # MARXEFE agent (internal objective unchanged)
    # Buffers: ubuffer holds `input_buffer` input vectors (current + Mu past),
    # ybuffer holds `output_buffer` past output vectors. Defaults: 3 inputs, 10
    # outputs.
    Mu  = int(input_buffer) - 1     # delay_inp -> ubuffer width = Mu+1 = input_buffer
    My  = int(output_buffer)        # delay_out -> ybuffer width = output_buffer
    Dy  = 4   # observation dim: vx, vy, pitch, roll
    Du  = 8
    len_horizon = int(time_horizon)   # EFE planning horizon (steps); default 2

    Nu0     = float(nu0)
    Omega0  = omega0_scale * np.diag(np.ones(Dy))
    reg_dim = Dy * My + Du * (Mu + 1)
    Lambda0 = lambda0_scale * np.diag(np.ones(reg_dim))
    Mean0   = 1e-8 * rnd.randn(reg_dim, Dy)

    mu_t      = 0.5 * (bounds_lower + bounds_upper)
    sigma_t   = control_prior_scale * (bounds_upper - bounds_lower) / (2.0 * n_sigma)
    upsilon_t = 1.0 / (sigma_t ** 2)
    Lambda_u  = torch.diag(upsilon_t)
    mu0       = mu_t.numpy()
    Upsilon0  = Lambda_u.numpy()

    # Goal prior — 4D: [vx, vy, pitch, roll]. Target forward velocity vx*,
    # zero lateral velocity, level attitude.
    sigma_vx, sigma_vy, sigma_pitch, sigma_roll = goal_prior_std
    m_star = np.array([target_velocity, 0.0, 0.0, 0.0])
    v_star = np.diag([sigma_vx**2, sigma_vy**2, sigma_pitch**2, sigma_roll**2])
    goal   = multivariate_normal(m_star, v_star)

    agent = MARXAgent(
        coefficients_mean_matrix    = Mean0.copy(),
        coefficients_row_covariance = Lambda0.copy(),
        precision_scale             = Omega0.copy(),
        precision_degrees           = Nu0,
        control_prior_mean          = mu0.copy(),
        control_prior_precision     = Upsilon0.copy(),
        goal_prior                  = goal,
        Dy=Dy, Du=Du, delay_inp=Mu, delay_out=My, time_horizon=len_horizon,
        forgetting=forgetting,
    )

    # Fresh interpolation state for this optimization run.
    global _prev_params_marx
    _prev_params_marx = None

    dtype        = torch.double
    train_X_orig = torch.empty(0, bounds.shape[1], dtype=dtype)
    train_Y      = torch.empty(0, 1, dtype=dtype)

    param_names = ["gc", "gp", "omegaswing", "omegastance",
                   "mu", "xofffront", "xoffhind", "sigmaN"]

    print("\n" + "=" * 70)
    print("MARXEFE OPTIMIZATION OF CPG PARAMETERS (8D)")
    print("=" * 70)
    print(f"  Target fwd vel  : {target_velocity} m/s  (agent goal AND objective J; lateral target = 0)")
    print(f"  Robot mass      : {robot_mass} kg")
    print(f"  Total trials    : {n_trials}")
    print(f"  Optimizer       : {optimizer_name},  Seed: {seed}")
    print(f"  Du              : {Du},  Dy: {Dy}")
    print(f"  Prior mu0       : {mu0}")
    print(f"  Results CSV     : {csv_path}")
    print("=" * 70)

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for trial_idx in range(1, n_trials + 1):
        agent.reset_buffer()

        is_debug_trial = debug_first_trial and trial_idx == 1
        debug_prefix = (os.path.join(figures_dir, f"{optimizer_name}_debug_trial1")
                        if is_debug_trial else None)

        J, metrics = evaluate_candidate(
            params_np               = None,
            target_velocity         = target_velocity,
            robot_mass              = robot_mass,
            optimizer_name          = optimizer_name,
            seed                    = seed,
            trial_idx               = trial_idx,
            agent                   = agent,
            update_every            = update_every,
            ramp_steps              = ramp_steps,
            debug                   = is_debug_trial,
            debug_save_prefix       = debug_prefix,
        )

        csv_writer.writerow(metrics)
        csv_file.flush()

        params8d = np.array([
            metrics["gc"],          metrics["gp"],         metrics["omegaswing"],
            metrics["omegastance"], metrics["mu"],
            metrics["xofffront"],   metrics["xoffhind"],   metrics["sigmaN"],
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

    # Expose the trained agent so callers can run extra evaluation episodes
    # (e.g. for transition-recovery analysis) without re-training.
    global _last_agent
    _last_agent = agent

    return train_X_orig, train_Y, best_params


# =============================================================================
# PLOTTING — mirrors BO Figures 1 / 2 / 3
# =============================================================================

def plot_marxefe_debug(trial_data: dict, save_prefix: str) -> None:
    """Debug figure: posterior evolution + one-step predictive vs observation
    + control trajectory, within a single trial. Requires debug=True run."""
    required = ["t", "y_history", "preds_m", "preds_v", "u_history",
                "post_M_norm", "post_L_logdet", "post_O_tr", "post_O_logdet",
                "post_nu"]
    missing = [k for k in required if k not in trial_data]
    if missing:
        print(f"[plot_marxefe_debug] trial_data missing {missing}; skipping.")
        return

    t   = trial_data["t"]
    y   = trial_data["y_history"]       # (4, T)
    mu  = trial_data["preds_m"]         # (4, T)
    v   = trial_data["preds_v"]         # (4, T)
    uh  = trial_data["u_history"]       # (8, T)
    g   = trial_data.get("goals_m")     # (4, T) or None
    std = np.sqrt(np.clip(v, 0, None))

    trans = trial_data.get("transition_duration", None)

    # Figure A: posterior summaries + predictive
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle("MARXEFE debug: posterior + posterior predictive (trial 1)",
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t, trial_data["post_M_norm"], color='C0')
    ax.set(title=r"$\|M\|_F$ (AR coefficient posterior mean)",
           xlabel="t [s]", ylabel=r"$\|M\|_F$")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, trial_data["post_nu"], color='C1')
    ax.set(title=r"$\nu$ (precision df, should grow linearly)",
           xlabel="t [s]", ylabel=r"$\nu$")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, trial_data["post_L_logdet"], color='C2')
    ax.set(title=r"$\log|\Lambda|$ (regressor precision)",
           xlabel="t [s]", ylabel=r"$\log|\Lambda|$")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, trial_data["post_O_logdet"], color='C3', label=r"$\log|\Omega|$")
    ax2 = ax.twinx()
    ax2.plot(t, trial_data["post_O_tr"], color='C4', alpha=0.7,
             label=r"$\mathrm{tr}(\Omega)$")
    ax.set(title=r"$\Omega$ (output precision scale)", xlabel="t [s]",
           ylabel=r"$\log|\Omega|$")
    ax2.set_ylabel(r"$\mathrm{tr}(\Omega)$")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')

    obs_labels = ["pos_x (forward +Y) [m]", "pos_y (lateral X) [m]",
                  "pitch [rad]", "roll [rad]"]
    row_col = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (r, c) in enumerate(row_col):
        ax = axes[r, c]
        ax.plot(t, y[i],  color='steelblue', lw=1.5, label="observation")
        ax.plot(t, mu[i], color='crimson', lw=1.2, ls='--', label=r"pred $\mu_t$")
        ax.fill_between(t, mu[i] - 2 * std[i], mu[i] + 2 * std[i],
                        color='crimson', alpha=0.15, label=r"pred $\pm 2\sigma$")
        if g is not None:
            ax.plot(t, g[i], color='green', ls=':', lw=1, label="goal mean")
        if trans is not None:
            ax.axvline(trans, color='gray', ls=':', alpha=0.7)
        ax.set(title=obs_labels[i], xlabel="t [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    outA = f"{save_prefix}_posterior_and_predictive.png"
    fig.savefig(outA, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {outA}")

    # Figure B: control trajectory (8D CPG params over time)
    param_labels = ["g_c (clearance)", "g_p (penetration)", "omega_swing", "omega_stance",
                    "mu (amplitude)", "x_off_front", "x_off_hind", "sigma_N (Tegotae)"]
    fig2, axes2 = plt.subplots(4, 2, figsize=(16, 11))
    fig2.suptitle("MARXEFE debug: CPG parameter trajectory within trial 1",
                  fontsize=14, fontweight='bold')
    for i, lbl in enumerate(param_labels):
        ax = axes2.flat[i]
        ax.plot(t, uh[i], color='C0', lw=1.3)
        ax.axhline(bounds_lower[i].item(), color='gray', ls=':', lw=1, alpha=0.6)
        ax.axhline(bounds_upper[i].item(), color='gray', ls=':', lw=1, alpha=0.6)
        if trans is not None:
            ax.axvline(trans, color='gray', ls=':', alpha=0.7)
        ax.set(title=lbl, xlabel="t [s]", ylabel=lbl)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    outB = f"{save_prefix}_controls.png"
    fig2.savefig(outB, dpi=150)
    plt.close(fig2)
    print(f"✅ Saved {outB}")


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

    param_names  = ["gc", "gp", "omegaswing", "omegastance",
                    "mu", "xofffront", "xoffhind", "sigmaN"]
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
    param_labels = ["g_c (clearance)", "g_p (penetration)", "omega_swing", "omega_stance",
                    "mu (amplitude)", "x_off_front", "x_off_hind", "sigma_N (Tegotae)"]
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
