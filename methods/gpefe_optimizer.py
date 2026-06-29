"""GPEFE active-inference optimizer for CPG parameters.

Replaces the linear MARX generative model in MARXAgent with a Gaussian-process
auto-regressive (GP-AR) model: independent per-output ARD-RBF GPs over the
augmented regressor x_k = [u_k, ..., u_{k-Mu}, y_{k-1}, ..., y_{k-My}].

API parity with MARXAgent — `update`, `posterior_predictive`, `EFE`,
`minimizeEFE`, `reset_buffer` — so the driver layer is unchanged in spirit.
The simulation/PyBullet helpers, the objective J, and the plot pipeline are
reused from `methods.marxefe_optimizer`.

See `methods/GPEFE_derivations.md` for the math.
"""

import csv
import os
import pickle
import time

import casadi as ca
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pybullet as p
import scipy.linalg as sla
import torch
from scipy.stats import multivariate_normal

from methods.cpg_bounds import bounds, bounds_lower, bounds_upper
from methods import marxefe_optimizer as mxo
from methods.marxefe_optimizer import (
    SWING_ENTER, SWING_EXIT, STANCE_ENTER, STANCE_EXIT,
    n_sigma, get_phase, compute_feedback_u,
    load_environment, load_robot, reset_simulation,
    extract_observation, check_if_fallen,
    compute_objective,
)


# =============================================================================
# GP AGENT — active inference with a Gaussian-process auto-regressive model.
# =============================================================================

class _ExactARDGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPAgent:
    """Active inference agent with independent per-output ARD-RBF GPs."""

    def __init__(self,
                 control_prior_mean,
                 control_prior_precision,
                 goal_prior,
                 Dy=4,
                 Du=8,
                 delay_inp=2,
                 delay_out=4,
                 time_horizon=5,
                 window_size=500,
                 sigma_n0=0.05,
                 sigma_f0=1.0,
                 ell0=2.0,
                 refit_each_trial=True,
                 recompute_every=10,
                 jitter=1e-5,
                 fit_iters=50):

        self.Dy = Dy
        self.Du = Du
        self.Dx = Du * (delay_inp + 1) + Dy * delay_out
        self.delay_inp = delay_inp
        self.delay_out = delay_out
        self.thorizon  = time_horizon

        self.μ = np.asarray(control_prior_mean, dtype=float)
        self.Υ = np.asarray(control_prior_precision, dtype=float)
        self.goal_prior = goal_prior

        self.ubuffer = np.zeros((Du, delay_inp + 1))
        self.ybuffer = np.zeros((Dy, delay_out))

        # Sliding-window training set (full input space, all data ever seen
        # is folded into the most recent window_size points)
        self.window_size = int(window_size)
        self.X_train = np.zeros((0, self.Dx))
        self.Y_train = np.zeros((0, Dy))

        # Per-output hyperparameters (in log space): sigma_f, ell (Dx-vector), sigma_n
        self.log_sigma_f = np.full(Dy, np.log(sigma_f0))
        self.log_ell     = np.full((Dy, self.Dx), np.log(ell0))
        self.log_sigma_n = np.full(Dy, np.log(sigma_n0))

        # Cached posterior tensors per output (set by _recompute_posterior)
        self.L_chol = [None] * Dy
        self.alpha  = [None] * Dy
        self._posterior_dirty = True
        self._steps_since_refit = 0

        # Input standardisation — fit once after first trial
        self.x_mean = np.zeros(self.Dx)
        self.x_std  = np.ones(self.Dx)
        self.normalized = False

        self.refit_each_trial = bool(refit_each_trial)
        self.recompute_every  = int(recompute_every)
        self.jitter   = float(jitter)
        self.fit_iters = int(fit_iters)

        self.free_energy = float('inf')

    # ----- buffer / book-keeping --------------------------------------------

    def backshift(self, x, a):
        if x.ndim == 2:
            return np.column_stack((a, x[:, :-1]))
        elif x.ndim == 1:
            N = x.size
            S = np.eye(N, k=-1)
            e = np.zeros(N)
            e[0] = 1.0
            return S.dot(x) + e * a

    def reset_buffer(self):
        self.ubuffer = np.zeros((self.Du, self.delay_inp + 1))
        self.ybuffer = np.zeros((self.Dy, self.delay_out))
        self.free_energy = float('inf')

        # Refit hyperparameters (and posterior) from accumulated cross-trial data
        if self.refit_each_trial and self.X_train.shape[0] >= 30:
            self._fit_hyperparameters()
            self._recompute_posterior()
            self._posterior_dirty = False
            self._steps_since_refit = 0

    # ----- update ------------------------------------------------------------

    def update(self, y_k, u_k):
        self.ubuffer = self.backshift(self.ubuffer, u_k)
        x_k = np.concatenate([self.ubuffer.flatten(), self.ybuffer.flatten()])

        self.X_train = np.vstack([self.X_train, x_k[None, :]])
        self.Y_train = np.vstack([self.Y_train, np.asarray(y_k)[None, :]])
        if self.X_train.shape[0] > self.window_size:
            self.X_train = self.X_train[-self.window_size:]
            self.Y_train = self.Y_train[-self.window_size:]

        self._steps_since_refit += 1
        self._posterior_dirty = True
        if self._steps_since_refit >= self.recompute_every:
            self._recompute_posterior()
            self._posterior_dirty = False
            self._steps_since_refit = 0

        self.ybuffer = self.backshift(self.ybuffer, y_k)

    # ----- kernel ------------------------------------------------------------

    def _normalise(self, X):
        return (X - self.x_mean) / self.x_std if self.normalized else X

    def _kernel_matrix(self, X1, X2, d):
        ell = np.exp(self.log_ell[d])
        sf2 = np.exp(2 * self.log_sigma_f[d])
        Xs1 = X1 / ell
        Xs2 = X2 / ell
        d2 = (np.sum(Xs1 ** 2, axis=1, keepdims=True)
              - 2 * Xs1 @ Xs2.T
              + np.sum(Xs2 ** 2, axis=1)[None, :])
        d2 = np.clip(d2, 0, None)
        return sf2 * np.exp(-0.5 * d2)

    def _recompute_posterior(self):
        N = self.X_train.shape[0]
        if N == 0:
            self.L_chol = [None] * self.Dy
            self.alpha  = [None] * self.Dy
            return
        Xn = self._normalise(self.X_train)
        for d in range(self.Dy):
            K = self._kernel_matrix(Xn, Xn, d)
            sn2 = float(np.exp(2 * self.log_sigma_n[d]))
            A = K + (sn2 + self.jitter) * np.eye(N)
            try:
                L = np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(A + 1e-3 * np.eye(N))
            y = self.Y_train[:, d]
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.L_chol[d] = L
            self.alpha[d]  = alpha

    # ----- posterior predictive ---------------------------------------------

    def posterior_predictive(self, x_t):
        """Return (eta, mu, Psi) compatible with MARX driver's recording.

        For the GP the predictive is exactly Gaussian, so we set
        eta -> 1e12 (≈ infinity) and Psi = diag(1/s²_d). Then
        inv(Psi) * eta/(eta-2) ≈ diag(s²_d), which is what the driver
        reports in `preds_v`.
        """
        N = self.X_train.shape[0]
        Dy = self.Dy
        if N == 0 or self.L_chol[0] is None:
            mu = np.zeros(Dy)
            s2 = np.array([np.exp(2 * self.log_sigma_f[d])
                           + np.exp(2 * self.log_sigma_n[d])
                           for d in range(Dy)])
        else:
            xn = self._normalise(np.asarray(x_t).reshape(1, -1))
            Xn = self._normalise(self.X_train)
            mu = np.zeros(Dy)
            s2 = np.zeros(Dy)
            for d in range(Dy):
                k_star = self._kernel_matrix(Xn, xn, d).ravel()
                k_ss   = float(np.exp(2 * self.log_sigma_f[d]))
                v      = np.linalg.solve(self.L_chol[d], k_star)
                mu[d]  = float(k_star @ self.alpha[d])
                s2[d]  = max(k_ss - float(v @ v)
                             + float(np.exp(2 * self.log_sigma_n[d])), 1e-10)
        eta = 1e12
        Psi = np.diag(1.0 / s2)
        return eta, mu, Psi

    def predictions(self, controls, time_horizon=1):
        m_y = np.zeros((self.Dy, time_horizon))
        S_y = np.zeros((self.Dy, self.Dy, time_horizon))
        ybuffer = self.ybuffer.copy()
        ubuffer = self.ubuffer.copy()
        for t in range(time_horizon):
            ubuffer = self.backshift(ubuffer, controls[:, t])
            x_t = np.concatenate([ubuffer.flatten(), ybuffer.flatten()])
            _, mu, Psi = self.posterior_predictive(x_t)
            m_y[:, t]    = mu
            S_y[:, :, t] = np.linalg.inv(Psi)
            ybuffer = self.backshift(ybuffer, mu)
        return m_y, S_y

    # ----- EFE (numpy, used for debugging only) -----------------------------

    def EFE(self, controls):
        ybuffer = self.ybuffer.copy()
        ubuffer = self.ubuffer.copy()
        m_star = np.asarray(self.goal_prior.mean)
        inv_S_star = np.linalg.inv(self.goal_prior.cov)
        J = 0.0
        for t in range(self.thorizon):
            u_t = controls[t * self.Du:(t + 1) * self.Du]
            ubuffer = self.backshift(ubuffer, u_t)
            x_t = np.concatenate([ubuffer.flatten(), ybuffer.flatten()])
            _, mu, Psi = self.posterior_predictive(x_t)
            s2 = 1.0 / np.diag(Psi)
            mi = 0.5 * np.sum(np.log(s2))
            ce = 0.5 * (np.sum(np.diag(inv_S_star) * s2)
                        + (mu - m_star) @ inv_S_star @ (mu - m_star))
            cp = 0.5 * (u_t - self.μ) @ self.Υ @ (u_t - self.μ)
            J += mi + ce + cp
            ybuffer = self.backshift(ybuffer, mu)
        return J

    # ----- minimiseEFE (CasADi/IPOPT) ---------------------------------------

    def minimizeEFE(self, u_0=None, verbose=False, control_lims=(-np.inf, np.inf),
                    lambda_energy=0.0, max_iter=200, tol=1e-6):
        Du = self.Du
        Dy = self.Dy
        thorizon = self.thorizon
        n_u = thorizon * Du

        if self._posterior_dirty:
            self._recompute_posterior()
            self._posterior_dirty = False
            self._steps_since_refit = 0

        N = self.X_train.shape[0]

        if u_0 is None:
            u_0 = np.tile(self.μ, thorizon)
        u_0 = np.asarray(u_0, dtype=float).reshape(-1)
        if u_0.size != n_u:
            u_0 = np.tile(self.μ, thorizon)

        if (isinstance(control_lims, tuple) and len(control_lims) == 2
                and np.isscalar(control_lims[0])):
            lbx = np.full(n_u, float(control_lims[0]))
            ubx = np.full(n_u, float(control_lims[1]))
        else:
            lbx = np.array([b[0] for b in control_lims], dtype=float)
            ubx = np.array([b[1] for b in control_lims], dtype=float)

        if N == 0 or self.L_chol[0] is None:
            return u_0

        # ---- Per-output precomputations as ca.DM constants ----
        Xn      = self._normalise(self.X_train)            # (N, Dx)
        x_mean_dm = ca.DM(self.x_mean.reshape(-1, 1))
        x_std_dm  = ca.DM(self.x_std.reshape(-1, 1))

        per_d = []
        for d in range(Dy):
            ell = np.exp(self.log_ell[d])                  # (Dx,)
            sf2 = float(np.exp(2 * self.log_sigma_f[d]))
            sn2 = float(np.exp(2 * self.log_sigma_n[d]))
            Xn_scaled = Xn / ell                           # (N, Dx)
            sq_X = np.sum(Xn_scaled ** 2, axis=1).reshape(-1, 1)  # (N,1)
            L_inv = sla.solve_triangular(
                self.L_chol[d], np.eye(N), lower=True)     # (N,N)
            per_d.append({
                'ell':       ca.DM(ell.reshape(-1, 1)),
                'sf2':       sf2,
                'sn2':       sn2,
                'Xn_scaled': ca.DM(Xn_scaled),
                'sq_X':      ca.DM(sq_X),
                'L_inv':     ca.DM(L_inv),
                'alpha':     ca.DM(self.alpha[d].reshape(-1, 1)),
            })

        m_star_dm     = ca.DM(np.asarray(self.goal_prior.mean).reshape(-1, 1))
        inv_S_star    = np.linalg.inv(self.goal_prior.cov)
        inv_S_star_dm = ca.DM(inv_S_star)
        inv_S_star_diag_dm = ca.DM(np.diag(inv_S_star).reshape(-1, 1))
        mu_prior_dm   = ca.DM(self.μ.reshape(-1, 1))
        Upsilon_dm    = ca.DM(self.Υ)

        ybuf = ca.DM(self.ybuffer)
        ubuf = ca.DM(self.ubuffer)

        u_sym = ca.MX.sym('u', n_u)
        J_sym = ca.MX(0)

        for t in range(thorizon):
            u_t = u_sym[t * Du:(t + 1) * Du]
            ubuf = ca.horzcat(u_t, ubuf[:, :-1])
            x_t = ca.vertcat(ca.reshape(ubuf.T, -1, 1),
                             ca.reshape(ybuf.T, -1, 1))
            x_t_n = (x_t - x_mean_dm) / x_std_dm if self.normalized else x_t

            mu_components = []
            s2_components = []
            for d in range(Dy):
                pd = per_d[d]
                xt_scaled = x_t_n / pd['ell']                 # (Dx,1) MX
                cross  = pd['Xn_scaled'] @ xt_scaled          # (N,1)  MX
                sq_xt  = xt_scaled.T @ xt_scaled              # (1,1)  MX
                d2     = pd['sq_X'] - 2 * cross + sq_xt       # (N,1)  MX
                d2     = ca.fmax(d2, 0)
                k_star = pd['sf2'] * ca.exp(-0.5 * d2)        # (N,1)  MX
                mu_d   = ca.dot(k_star, pd['alpha'])
                v      = pd['L_inv'] @ k_star                 # (N,1)  MX
                s2_d   = pd['sf2'] - ca.dot(v, v) + pd['sn2']
                s2_d   = ca.fmax(s2_d, 1e-8)
                mu_components.append(mu_d)
                s2_components.append(s2_d)

            mu_vec = ca.vertcat(*mu_components)               # (Dy,1)
            s2_vec = ca.vertcat(*s2_components)               # (Dy,1)

            mi   = 0.5 * ca.sum1(ca.log(s2_vec))
            diff = mu_vec - m_star_dm
            ce   = 0.5 * (ca.dot(inv_S_star_diag_dm, s2_vec)
                          + diff.T @ inv_S_star_dm @ diff)
            up_diff = u_t - mu_prior_dm
            cp = 0.5 * (up_diff.T @ Upsilon_dm @ up_diff)

            J_sym = J_sym + mi + ce + cp
            if lambda_energy != 0.0:
                J_sym = J_sym + lambda_energy * (u_t.T @ u_t)

            ybuf = ca.horzcat(mu_vec, ybuf[:, :-1])

        nlp  = {'x': u_sym, 'f': J_sym}
        opts = {
            'print_time': False,
            'ipopt.print_level': 5 if verbose else 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': int(max_iter),
            'ipopt.tol': float(tol),
            'ipopt.hessian_approximation': 'limited-memory',
        }
        solver = ca.nlpsol('gpefe_solver', 'ipopt', nlp, opts)
        sol = solver(x0=u_0, lbx=lbx, ubx=ubx)
        return np.array(sol['x']).reshape(-1)

    # ----- hyperparameter fitting (gpytorch / L-BFGS) -----------------------

    def _fit_hyperparameters(self):
        N = self.X_train.shape[0]
        if N < 5:
            return

        # Set input standardisation from accumulated data (once, persistent)
        if not self.normalized and N >= 30:
            self.x_mean = self.X_train.mean(axis=0)
            self.x_std  = self.X_train.std(axis=0) + 1e-8
            self.normalized = True

        Xn = self._normalise(self.X_train)
        Xn_t = torch.tensor(Xn, dtype=torch.float64)

        for d in range(self.Dy):
            y_t = torch.tensor(self.Y_train[:, d], dtype=torch.float64)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
            model = _ExactARDGP(Xn_t, y_t, likelihood, self.Dx).double()

            # Warm-start from current values
            with torch.no_grad():
                model.likelihood.noise = max(
                    float(np.exp(2 * self.log_sigma_n[d])), 1e-4)
                model.covar_module.outputscale = float(
                    np.exp(2 * self.log_sigma_f[d]))
                model.covar_module.base_kernel.lengthscale = torch.tensor(
                    np.exp(self.log_ell[d]), dtype=torch.float64
                ).unsqueeze(0)

            model.train(); likelihood.train()
            optim = torch.optim.LBFGS(
                model.parameters(), lr=0.1, max_iter=self.fit_iters,
                tolerance_grad=1e-5, tolerance_change=1e-7,
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            def closure():
                optim.zero_grad()
                out  = model(Xn_t)
                loss = -mll(out, y_t)
                loss.backward()
                return loss

            try:
                optim.step(closure)
            except Exception as e:
                print(f"[GPAgent] hyperparameter fit failed for output {d}: {e}")
                continue

            with torch.no_grad():
                self.log_sigma_f[d] = 0.5 * float(
                    torch.log(model.covar_module.outputscale.detach()))
                ell = model.covar_module.base_kernel.lengthscale.detach().squeeze().cpu().numpy()
                self.log_ell[d]     = np.log(np.clip(ell, 1e-3, 1e3))
                self.log_sigma_n[d] = 0.5 * float(
                    torch.log(model.likelihood.noise.detach()))

    # ----- pickle helpers ---------------------------------------------------

    def save_agent(self, filename, makedir=False):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            if makedir:
                os.makedirs(directory)
            else:
                raise FileNotFoundError(f"Directory {directory} does not exist.")
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load_agent(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)


# =============================================================================
# GPEFE EPISODE (parallels run_episode_maxrefe; GP-specific debug recording)
# =============================================================================

def run_episode_gpefe(agent, robot, joint_IDs_full_arg,
                      filtered_joint_IDs_arg, feet_joint_IDs_arg,
                      dt, episode_length=4.5,
                      lambda_energy=1e-2,
                      target_forward_position=4.0,
                      debug=False):
    """One GPEFE-controlled episode. Mirrors run_episode_maxrefe."""
    ori_default = [0.0, 0.5, 0.5, 0.0]

    cpg_x, cpg_y = reset_simulation(p, robot, filtered_joint_IDs_arg, ori_default)

    n_legs = 4
    y_k, base_pos, base_orientation = extract_observation(p, robot, ori_default)

    mxo.phase_state_memory = []
    for j in range(n_legs):
        if   cpg_y[j] > SWING_ENTER:
            mxo.phase_state_memory.append('swing')
        elif cpg_y[j] < STANCE_ENTER:
            mxo.phase_state_memory.append('stance')
        else:
            mxo.phase_state_memory.append('transition')

    coupling_gain  = agent.μ[0]
    w_swing        = agent.μ[1]
    w_stance       = agent.μ[2]
    F_FAST         = agent.μ[3]
    STOP_GAIN      = agent.μ[4]
    hip_amplitude  = agent.μ[5]
    knee_amplitude = agent.μ[6]
    b              = agent.μ[7]

    alpha       = 3.0
    beta        = 12.0
    u_lim       = 2.0
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

    action_update_frequency = 1
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

    if debug:
        post_train_size = np.zeros(num_steps)
        post_logsf_mean = np.zeros(num_steps)
        post_logell_mean = np.zeros(num_steps)
        post_logsn_mean = np.zeros(num_steps)

    n_joints     = 2 * n_legs
    torques_log  = np.zeros((n_joints, num_steps))
    qdot_log     = np.zeros((n_joints, num_steps))
    base_pos_log = np.zeros((num_steps, 3))
    vx_log       = np.zeros(num_steps)
    vy_log       = np.zeros(num_steps)

    policy = np.tile(agent.μ, agent.thorizon) + 0.05 * rnd.randn(agent.Du * agent.thorizon)
    is_fallen = False

    leg_names      = ["FL", "FR", "RL", "RR"]
    hip_joint_ids  = [1, 5, 9, 13]
    knee_joint_ids = [2, 6, 10, 14]

    DEBOUNCE_THRESHOLD   = 2
    debounced_contacts   = np.zeros(n_legs, dtype=int)
    contact_change_count = np.zeros(n_legs, dtype=int)

    for k_step in range(num_steps):
        t = k_step * dt
        times[k_step]      = t
        y_history[:, k_step] = y_k
        goals_m[:, k_step]   = agent.goal_prior.mean

        x_k = np.concatenate([agent.ubuffer.flatten(), agent.ybuffer.flatten()])
        eta_k, mu_k, Psi_k = agent.posterior_predictive(x_k)
        preds_m[:, k_step] = mu_k
        preds_v[:, k_step] = np.diag(np.linalg.inv(Psi_k) * eta_k / (eta_k - 2))

        params_8d = np.array([
            coupling_gain, w_swing, w_stance, F_FAST,
            STOP_GAIN, hip_amplitude, knee_amplitude, b
        ])
        params_8d = np.clip(params_8d, bounds_lower.numpy(), bounds_upper.numpy())
        coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN, \
            hip_amplitude, knee_amplitude, b = params_8d

        w_vec = np.zeros(n_legs)
        r_vec = np.zeros(n_legs)
        for j in range(n_legs):
            y_prev = cpg_y[j]
            x_prev = cpg_x[j]
            w = (w_stance / (np.exp(-b * y_prev) + 1)
                 + w_swing / (np.exp( b * y_prev) + 1))
            w_vec[j] = w
            r        = np.sqrt(x_prev ** 2 + y_prev ** 2)
            r_vec[j] = r
            cpg_x[j] += dt * (alpha * (u_lim - r ** 2) * x_prev - w * y_prev)

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
            coupling_gain=coupling_gain, STOP_gain=STOP_GAIN,
        )

        for j in range(n_legs):
            y_prev        = cpg_y[j]
            x_prev        = cpg_x[j]
            r             = r_vec[j]
            w             = w_vec[j]
            coupling_term = coupling_gain * np.dot(k_matrix[j, :], cpg_y)
            cpg_y[j] += dt * (beta * (u_lim - r ** 2) * y_prev
                              + w * x_prev + coupling_term + u_fb[j])

        for j in range(n_legs):
            leg_name                       = leg_names[j]
            abd_joint, hip_joint, kn_joint = joint_IDs_full_arg[leg_name]
            p.setJointMotorControl2(
                robot, abd_joint, p.POSITION_CONTROL,
                targetPosition=0.0, force=500
            )
            hip_angle  = hip_offset + hip_amplitude  * cpg_x[j]
            knee_angle = knee_offset - knee_amplitude * max(0, cpg_y[j])
            p.setJointMotorControl2(robot, hip_joint, p.POSITION_CONTROL, hip_angle)
            p.setJointMotorControl2(robot, kn_joint,  p.POSITION_CONTROL, knee_angle)

        p.stepSimulation()

        joint_idx = 0
        for jid_hip, jid_knee in zip(hip_joint_ids, knee_joint_ids):
            hs = p.getJointState(robot, jid_hip)
            ks = p.getJointState(robot, jid_knee)
            torques_log[joint_idx,     k_step] = hs[3]
            torques_log[joint_idx + 1, k_step] = ks[3]
            qdot_log[joint_idx,        k_step] = hs[1]
            qdot_log[joint_idx + 1,    k_step] = ks[1]
            joint_idx += 2

        y_k_new, base_pos, base_orientation = extract_observation(
            p, robot, ori_default
        )
        positions.append(np.array(base_pos))
        base_pos_log[k_step, :] = base_pos

        base_vel, _ = p.getBaseVelocity(robot)
        vx_log[k_step] = base_vel[1]
        vy_log[k_step] = base_vel[0]

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

        if debug:
            post_train_size[k_step] = float(agent.X_train.shape[0])
            post_logsf_mean[k_step] = float(np.mean(agent.log_sigma_f))
            post_logell_mean[k_step] = float(np.mean(agent.log_ell))
            post_logsn_mean[k_step] = float(np.mean(agent.log_sigma_n))

        if k_step > 0 and k_step % action_update_frequency == 0:
            bounds_agent = []
            for _ in range(agent.thorizon):
                for i in range(8):
                    bounds_agent.append((bounds_lower[i].item(), bounds_upper[i].item()))
            try:
                u_opt = agent.minimizeEFE(
                    u_0           = policy,
                    control_lims  = bounds_agent,
                    lambda_energy = lambda_energy,
                    max_iter      = 100,
                    tol           = 1e-4,
                )
                policy = u_opt
            except Exception as e:
                print(f"Warning: GP-EFE minimization failed at step {k_step}: {e}")
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
            if debug:
                y_history       = y_history[:, :actual_steps]
                u_history       = u_history[:, :actual_steps]
                preds_m         = preds_m[:, :actual_steps]
                preds_v         = preds_v[:, :actual_steps]
                goals_m         = goals_m[:, :actual_steps]
                post_train_size = post_train_size[:actual_steps]
                post_logsf_mean = post_logsf_mean[:actual_steps]
                post_logell_mean = post_logell_mean[:actual_steps]
                post_logsn_mean = post_logsn_mean[:actual_steps]
            num_steps = actual_steps
            break

    # POST-EPISODE METRICS
    start_idx = transition_steps
    distance  = np.linalg.norm(base_pos_log[-1, :2] - base_pos_log[0, :2])

    if len(roll_angles) > start_idx:
        roll_w  = roll_angles[start_idx:]
        pitch_w = pitch_angles[start_idx:]
        rms_roll  = np.rad2deg(np.sqrt(np.mean(roll_w  ** 2)))
        rms_pitch = np.rad2deg(np.sqrt(np.mean(pitch_w ** 2)))
        combined_stability = np.sqrt(rms_roll ** 2 + rms_pitch ** 2)
    else:
        combined_stability = 1000.0

    if len(base_pos_log) > start_idx:
        forward_distance = base_pos_log[-1, 1] - base_pos_log[0, 1]
        lateral_drift    = abs(base_pos_log[-1, 0] - base_pos_log[0, 0])
        T_steady   = (len(base_pos_log) - start_idx) * dt
        mean_vx    = ((base_pos_log[-1, 1] - base_pos_log[start_idx, 1]) / T_steady
                      if T_steady > 0 else 0.0)
        torques_s  = torques_log[:, start_idx:].T
        qdot_s     = qdot_log[:,   start_idx:].T
        mech_pwr   = np.sum(np.abs(torques_s * qdot_s)) * dt
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

    params_8d = np.array([
        coupling_gain, w_swing, w_stance, F_FAST,
        STOP_GAIN, hip_amplitude, knee_amplitude, b
    ])

    trial_data = {
        "t":                   times,
        "pos_x":               base_pos_log[:, 1],
        "pos_y":               base_pos_log[:, 0],
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
    if debug:
        trial_data.update({
            "y_history":        y_history,
            "u_history":        u_history,
            "preds_m":          preds_m,
            "preds_v":          preds_v,
            "goals_m":          goals_m,
            "post_train_size":  post_train_size,
            "post_logsf_mean":  post_logsf_mean,
            "post_logell_mean": post_logell_mean,
            "post_logsn_mean":  post_logsn_mean,
        })
    return trial_data


# =============================================================================
# EVALUATE_CANDIDATE
# =============================================================================

# Module-level robot handles (initialised once per Python session)
quadruped          = None
joint_IDs_full     = None
filtered_joint_IDs = None
feet_joint_IDs     = None


def evaluate_candidate_gp(params_np, target_forward_position, robot_mass,
                          optimizer_name, seed, trial_idx, agent=None,
                          debug=False, debug_save_prefix=None):
    sim_start = time.time()
    if agent is None:
        raise NotImplementedError("evaluate_candidate_gp requires an agent.")

    trial_data = run_episode_gpefe(
        agent, quadruped, joint_IDs_full, filtered_joint_IDs, feet_joint_IDs,
        dt=0.01, episode_length=4.5,
        lambda_energy=1e-2,
        target_forward_position=target_forward_position,
        debug=debug,
    )
    sim_time_sec = time.time() - sim_start

    if debug and debug_save_prefix is not None:
        plot_gpefe_debug(trial_data, debug_save_prefix)

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
        "stabilityindex": stability,
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
# GPEFE OPTIMISATION LOOP
# =============================================================================

def gpefe_optimize_cpg(bounds: torch.Tensor,
                       target_forward_position: float,
                       robot_mass: float,
                       n_trials: int = 10,
                       optimizer_name: str = "GPEFE",
                       seed: int = 0,
                       results_dir: str = "results",
                       debug_first_trial: bool = True,
                       goal_prior_std=(3.0, 2.0, np.deg2rad(45), np.deg2rad(45)),
                       control_prior_scale: float = 2.0,
                       window_size: int = 500,
                       recompute_every: int = 10,
                       refit_each_trial: bool = True,
                       sigma_f0: float = 1.0,
                       sigma_n0: float = 0.05,
                       ell0: float = 2.0,
                       fit_iters: int = 50,
                       ) -> tuple:
    """GPEFE optimisation loop, BO-compatible CSV schema."""
    global quadruped, joint_IDs_full, filtered_joint_IDs, feet_joint_IDs

    os.makedirs(results_dir, exist_ok=True)
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
        load_environment(dt, use_gui=False)
        (quadruped, _, joint_IDs_full,
         filtered_joint_IDs, feet_joint_IDs) = load_robot(p, robot_mass=robot_mass)
        print(f"\n✅ Environment initialized with robot ID: {quadruped} (mass={robot_mass} kg)")

    # GP-EFE agent setup
    Mu = 2
    My = 4
    Dy = 4
    Du = 8
    len_horizon = 5

    mu_t      = 0.5 * (bounds_lower + bounds_upper)
    sigma_t   = control_prior_scale * (bounds_upper - bounds_lower) / (2.0 * n_sigma)
    upsilon_t = 1.0 / (sigma_t ** 2)
    Lambda_u  = torch.diag(upsilon_t)
    mu0       = mu_t.numpy()
    Upsilon0  = Lambda_u.numpy()

    sigma_pos_x, sigma_pos_y, sigma_pitch, sigma_roll = goal_prior_std
    m_star = np.array([target_forward_position, 0.0, 0.0, 0.0])
    v_star = np.diag([sigma_pos_x ** 2, sigma_pos_y ** 2,
                      sigma_pitch  ** 2, sigma_roll  ** 2])
    goal   = multivariate_normal(m_star, v_star)

    agent = GPAgent(
        control_prior_mean      = mu0.copy(),
        control_prior_precision = Upsilon0.copy(),
        goal_prior              = goal,
        Dy=Dy, Du=Du, delay_inp=Mu, delay_out=My,
        time_horizon            = len_horizon,
        window_size             = window_size,
        recompute_every         = recompute_every,
        refit_each_trial        = refit_each_trial,
        sigma_f0                = sigma_f0,
        sigma_n0                = sigma_n0,
        ell0                    = ell0,
        fit_iters               = fit_iters,
    )

    dtype        = torch.double
    train_X_orig = torch.empty(0, bounds.shape[1], dtype=dtype)
    train_Y      = torch.empty(0, 1, dtype=dtype)

    param_names = ["couplinggain", "wswing", "wstance", "FFAST",
                   "STOPGAIN", "hipamplitude", "kneeamplitude", "b"]

    print("\n" + "=" * 70)
    print("GPEFE OPTIMIZATION OF CPG PARAMETERS (8D)")
    print("=" * 70)
    print(f"  Target fwd pos  : {target_forward_position} m")
    print(f"  Robot mass      : {robot_mass} kg")
    print(f"  Total trials    : {n_trials}")
    print(f"  Optimizer       : {optimizer_name},  Seed: {seed}")
    print(f"  Du              : {Du},  Dy: {Dy},  Dx: {agent.Dx}")
    print(f"  Window size     : {window_size}")
    print(f"  Recompute every : {recompute_every} steps")
    print(f"  Refit per trial : {refit_each_trial}")
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

        J, metrics = evaluate_candidate_gp(
            params_np               = None,
            target_forward_position = target_forward_position,
            robot_mass              = robot_mass,
            optimizer_name          = optimizer_name,
            seed                    = seed,
            trial_idx               = trial_idx,
            agent                   = agent,
            debug                   = is_debug_trial,
            debug_save_prefix       = debug_prefix,
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
            print(f"  Best so far: J={best_J:.4f} (trial {best_idx + 1})")

    csv_file.close()
    print(f"\n✅ CSV saved: {csv_path}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    best_J      = train_Y.max().item()
    best_idx    = train_Y.argmax().item()
    best_params = train_X_orig[best_idx].numpy()
    print(f"Best J = {best_J:.4f}  (trial {best_idx + 1})")
    print("Best parameters:")
    for name, value in zip(param_names, best_params):
        print(f"  {name:16s} = {value:.4f}")
    print("=" * 70)

    return train_X_orig, train_Y, best_params


# =============================================================================
# DEBUG PLOT
# =============================================================================

def plot_gpefe_debug(trial_data: dict, save_prefix: str) -> None:
    """Debug figure: GP hyperparameter / training-set trace + posterior
    predictive vs observation + control trajectory, within a single trial."""
    required = ["t", "y_history", "preds_m", "preds_v", "u_history",
                "post_train_size", "post_logsf_mean", "post_logell_mean",
                "post_logsn_mean"]
    missing = [k for k in required if k not in trial_data]
    if missing:
        print(f"[plot_gpefe_debug] trial_data missing {missing}; skipping.")
        return

    t   = trial_data["t"]
    y   = trial_data["y_history"]
    mu  = trial_data["preds_m"]
    v   = trial_data["preds_v"]
    uh  = trial_data["u_history"]
    g   = trial_data.get("goals_m")
    std = np.sqrt(np.clip(v, 0, None))
    trans = trial_data.get("transition_duration", None)

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle("GPEFE debug: GP state + posterior predictive (trial 1)",
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t, trial_data["post_train_size"], color='C0')
    ax.set(title="Training set size N",
           xlabel="t [s]", ylabel="N")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, trial_data["post_logsf_mean"], color='C1', label=r"$\overline{\log \sigma_f}$")
    ax.plot(t, trial_data["post_logsn_mean"], color='C2', label=r"$\overline{\log \sigma_n}$")
    ax.set(title="GP signal/noise scales (mean across outputs)",
           xlabel="t [s]", ylabel="log scale")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, trial_data["post_logell_mean"], color='C3')
    ax.set(title=r"Mean $\log \ell$ (lengthscale, across outputs and dims)",
           xlabel="t [s]", ylabel=r"$\overline{\log \ell}$")
    ax.grid(alpha=0.3)

    axes[1, 1].axis('off')

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
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    outA = f"{save_prefix}_posterior_and_predictive.png"
    fig.savefig(outA, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {outA}")

    param_labels = ["coupling_gain", "ω_swing", "ω_stance", "F_FAST",
                    "STOP_GAIN", "hip_amp", "knee_amp", "b"]
    fig2, axes2 = plt.subplots(4, 2, figsize=(16, 11))
    fig2.suptitle("GPEFE debug: CPG parameter trajectory within trial 1",
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


# =============================================================================
# RESULTS PLOT — reuses the MARXEFE plot code with a different label prefix
# =============================================================================

def plot_gpefe_results(csv_path: str,
                       target_forward_position: float = 4.0,
                       save_prefix: str = "gpefe") -> None:
    """Re-uses the MARXEFE plotting routine; same CSV schema, same panels."""
    from methods.marxefe_optimizer import plot_marxefe_results
    plot_marxefe_results(csv_path,
                         target_forward_position=target_forward_position,
                         save_prefix=save_prefix)
