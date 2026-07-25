"""Unified active-inference recovery agent (reactive GP + MARX dynamics belief).

A single active-inference agent that unifies three components under ONE Gaussian
goal prior over outputs y = [vx, vy, pitch, roll]:

  * a fast MatrixNormal-Wishart AR belief -- the existing
    ``marxefe_optimizer.MARXAgent`` reused UNMODIFIED -- updated every sim step
    (100 Hz), with the robot's measured joint angles as the exogenous input (a
    linear joint-angle -> output map) and past outputs as the autoregressive part;
  * a slow, event-triggered Gaussian-process map from the CPG control parameters
    to the outputs, with a persistent memory, that PROPOSES a recovery gait by
    minimizing Expected Free Energy under the goal prior;
  * a prediction-error TRIGGER: the cross-entropy from the MARX one-step posterior
    predictive to the goal prior, accumulated in a CUSUM. When the belief predicts
    outputs far from the goal (speed deficit / tilt), the accumulator crosses a
    bound and the agent fires -> propose a new gait.

The performance criterion for a gait is the log-probability of its observed output
under the goal prior. Trigger, control acquisition and scoring are all expressed
against the SAME goal prior, so this is a single free-energy-minimizing controller
rather than a detector bolted onto an optimizer.

DISTINCT from ``MARXAgent`` (which selects controls by minimizing EFE over the
MARX model itself and has no GP): here the MARX belief only drives the trigger,
and a reactive GP over controls selects the gait. The MARX object's own CasADi EFE
solver is never invoked -- only ``update`` / ``posterior_predictive`` /
``crossentropy``.
"""

import numpy as np


class UnifiedAIFAgent:
    mode = "aif"

    def __init__(self, incumbent, box, free_dims, seed=0, dt=0.01, *,
                 target_vx=0.5, goal_std=(0.25, 0.25, np.deg2rad(12), np.deg2rad(12)),
                 trigger_vx_std=1e3, trigger_vy_std=1e3, ar_order=2, forgetting=0.99,
                 n_init=4, pool=512, r_rep=0.04,
                 warmup_t=2.5, cusum_kappa=3.0, cusum_h=5.0):
        self.inc = np.asarray(incumbent, float)
        self.lo = np.asarray(box[0], float)
        self.hi = np.asarray(box[1], float)
        self.fd = list(free_dims)
        self.flo, self.fhi = self.lo[self.fd], self.hi[self.fd]
        self.d = len(self.fd)
        self.dt = float(dt)
        self.rng = np.random.default_rng(int(seed))
        self.n_init = int(n_init)
        self.pool = int(pool)
        self.r_rep = float(r_rep)

        # goal prior over y = [vx, vy, pitch, roll] (diagonal). Trigger and control
        # DECOUPLE on the LINEAR velocities: the CONTROL/EFE goal keeps a tight vx
        # (still drives "as fast as it can"), while the TRIGGER goal loosens both vx
        # AND vy (via trigger_vx_std / trigger_vy_std) so it is an UPRIGHTNESS-only
        # signal (pitch/roll) -- it quiets whenever the robot stays level, however
        # slow and however much it must crab sideways to compensate an asymmetric
        # fault. Pitch/roll stay tight and shared.
        control_std = np.asarray(goal_std, float)
        self.m_goal = np.array([target_vx, 0.0, 0.0, 0.0], float)
        self.goal_var = control_std ** 2                     # CONTROL goal (EFE)
        self.S_goal_inv = np.diag(1.0 / self.goal_var)

        # fast MARX belief (REUSED, unmodified): u = 8 joint angles, y = 4 outputs,
        # instantaneous exogenous map (delay_inp=0 via input_buffer=1) + AR order.
        # Its goal_prior is the TRIGGER goal (loose vx & vy -> pitch/roll only).
        tvx = float(trigger_vx_std) if trigger_vx_std is not None else float(control_std[0])
        tvy = float(trigger_vy_std) if trigger_vy_std is not None else float(control_std[1])
        trig_std = (tvx, tvy, control_std[2], control_std[3])
        from methods.marxefe_optimizer import build_marx_agent
        self.marx = build_marx_agent(
            target_velocity=float(target_vx),
            goal_prior_std=tuple(trig_std),
            input_buffer=1, output_buffer=int(ar_order), forgetting=float(forgetting))

        # GP memory: control (reduced, normalized), observed 4-D output, fell flag
        self.Xn, self.Yout, self.Fell = [], [], []

        # trigger CUSUM state (agent-owned)
        self.warmup_steps = int(round(warmup_t / self.dt))
        self.cusum_kappa = float(cusum_kappa)
        self.cusum_h = float(cusum_h)
        self._n = 0
        self._warm = []
        self.H_base = None
        self.H_scale = 1.0
        self.armed = False
        self.S = 0.0
        self.H = 0.0            # last goal cross-entropy (the trigger signal)
        self._last_H = 0.0
        self.surprise = 0.0     # last model surprise (diagnostic, self-quieting)
        self._last_surprise = 0.0
        self.ce_mean = np.zeros(4)   # per-dim cross-entropy mean-terms (diagnostic)
        self.ce_trace = 0.0          # cross-entropy covariance-trace term

    # ── reduced <-> full control ─────────────────────────────────────────────
    def _norm(self, full):
        return (np.asarray(full, float)[self.fd] - self.flo) / (self.fhi - self.flo)

    def _expand(self, xr_norm):
        full = self.inc.copy()
        full[self.fd] = self.flo + np.clip(xr_norm, 0.0, 1.0) * (self.fhi - self.flo)
        return full

    # ── per-step: update belief + accumulate the cross-entropy trigger ───────
    def observe(self, y, u):
        """Assimilate one 100 Hz observation. `y`=[vx,vy,pitch,roll],
        `u`=measured joint angles (Du=8). Updates the MARX belief and the trigger
        CUSUM on cross-entropy(predictive || goal)."""
        y = np.asarray(y, float)
        u = np.asarray(u, float)
        # one-step predictive cross-entropy to the goal, on the SAME regressor the
        # update will use (backshift is non-mutating), BEFORE assimilating y.
        try:
            ubuf = self.marx.backshift(self.marx.ubuffer, u)
            x = np.concatenate([ubuf.flatten(), self.marx.ybuffer.flatten()])
            H = float(self.marx.crossentropy(x))
            if not np.isfinite(H):
                H = self._last_H
        except Exception:
            H = self._last_H
        self._last_H = self.H = H
        # decompose the trigger cross-entropy into per-dim mean-terms + a covariance
        # trace term (diagnostic: what keeps the stability trigger lit?).
        try:
            eta, mu, Psi = self.marx.posterior_predictive(x)
            Spred = np.linalg.inv(Psi) * eta / (eta - 2.0)
            dd = np.diag(self.marx.goal_prior.cov)          # trigger goal variances
            mg = self.marx.goal_prior.mean
            self.ce_mean = 0.5 * (mu - mg) ** 2 / dd        # [vx,vy,pitch,roll]
            self.ce_trace = 0.5 * float(np.sum(np.diag(Spred) / dd))
        except Exception:
            pass
        # model surprise: NLL of the observed output under the one-step predictive
        # (a self-quieting alternative trigger -- returns to baseline once the
        # belief re-tracks the perturbed dynamics). Diagnostic only; not the trigger.
        try:
            self.surprise = -float(self.marx.log_evidence(y, x))
            if not np.isfinite(self.surprise):
                self.surprise = self._last_surprise
        except Exception:
            self.surprise = self._last_surprise
        self._last_surprise = self.surprise
        try:
            self.marx.update(y, u)
        except Exception:
            pass

        self._n += 1
        if self._n <= self.warmup_steps:              # learn healthy baseline
            self._warm.append(H)
            if self._n == self.warmup_steps:
                w = np.array(self._warm[len(self._warm) // 2:])
                self.H_base = float(np.median(w))
                self.H_scale = float(max(np.std(w), 0.1 * abs(self.H_base), 1e-6))
                self.armed = True
            return
        if self.armed:                                # one-sided CUSUM on excess
            e = (H - self.H_base) / self.H_scale
            self.S = max(0.0, self.S + e - self.cusum_kappa)

    def should_fire(self):
        return bool(self.armed and self.S > self.cusum_h)

    def on_reset(self):
        """After a fall/heal: clear the accumulator and the AR lag buffers (so the
        fall transient does not linger as autoregressive input), keeping the learned
        coefficients and the healthy baseline."""
        self.S = 0.0
        try:
            self.marx.ybuffer[:] = 0.0
            self.marx.ubuffer[:] = 0.0
            self.marx._const = None
        except Exception:
            pass

    # ── GP over controls -> outputs, EFE acquisition ─────────────────────────
    def _fit_gps(self):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        X = np.array(self.Xn, float)
        Yo = np.array(self.Yout, float)                # (N, 4)
        Xt = torch.tensor(X, dtype=torch.double)
        models, stats = [], []
        for j in range(Yo.shape[1]):
            Yj = Yo[:, j]
            Ym, Ys = float(Yj.mean()), float(max(Yj.std(), 1e-6))
            Yt = torch.tensor(((Yj - Ym) / Ys)[:, None], dtype=torch.double)
            m = SingleTaskGP(Xt, Yt)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
            m.eval()
            noise = float(m.likelihood.noise.mean().item()) * Ys ** 2
            models.append(m)
            stats.append((Ym, Ys, max(noise, 1e-9)))
        return models, stats

    def _predict(self, models, stats, pool_u):
        import torch
        mu = np.zeros((len(pool_u), len(models)))
        sig = np.zeros_like(mu)
        Ut = torch.tensor(pool_u, dtype=torch.double)
        with torch.no_grad():
            for j, (m, (Ym, Ys, _)) in enumerate(zip(models, stats)):
                post = m.posterior(Ut)
                mu[:, j] = post.mean.squeeze(-1).numpy() * Ys + Ym
                sig[:, j] = np.sqrt(post.variance.squeeze(-1).numpy()) * Ys
        return mu, sig

    def propose(self):
        """Return a full CPG param vector selected by minimizing EFE (cross-entropy
        of the GP-predicted output to the goal prior, minus information gain) over a
        Sobol pool. Random init until `n_init` observations are gathered."""
        if len(self.Yout) < self.n_init:
            return self._expand(self.rng.uniform(0.0, 1.0, size=self.d)), "init"
        try:
            pool = self._sobol(self.pool)
            models, stats = self._fit_gps()
            mu, sig = self._predict(models, stats, pool)
            noise = np.array([s[2] for s in stats])                 # (4,)
            gvar = self.goal_var                                    # (4,)
            # EFE = pragmatic (cross-entropy to goal) - epistemic (info gain)
            pragmatic = 0.5 * np.sum(
                (sig ** 2 + noise + (mu - self.m_goal) ** 2) / gvar, axis=1)
            epistemic = 0.5 * np.sum(np.log1p(sig ** 2 / noise), axis=1)
            G = pragmatic - epistemic
            # never re-evaluate a point we already have
            if self.Xn:
                Xn = np.array(self.Xn, float)
                dmin = np.linalg.norm(pool[:, None, :] - Xn[None, :, :],
                                      axis=2).min(axis=1)
                G = np.where(dmin < self.r_rep, np.inf, G)
            idx = int(np.argmin(G))
            return self._expand(pool[idx]), "efe"
        except Exception as e:
            print(f"[aif] GP EFE failed ({e}); random fallback", flush=True)
            return self._expand(self.rng.uniform(0.0, 1.0, size=self.d)), "fallback"

    def _sobol(self, n):
        try:
            from scipy.stats.qmc import Sobol
            m = int(np.ceil(np.log2(max(2, n))))
            u = Sobol(d=self.d, seed=int(self.rng.integers(1 << 30))).random_base2(m)
            return u[:n]
        except Exception:
            return self.rng.uniform(0, 1, size=(n, self.d))

    def update(self, cand, y_obs, fell):
        """Fold an event outcome into the GP memory. `y_obs` = mean observed output
        [vx,vy,pitch,roll] over the post-adaptation window (or near the fall)."""
        self.Xn.append(self._norm(cand))
        self.Yout.append(np.asarray(y_obs, float))
        self.Fell.append(int(bool(fell)))

    def goal_logprob(self, y_obs):
        """Performance criterion: log N(y_obs | goal prior) (up to a constant)."""
        d = np.asarray(y_obs, float) - self.m_goal
        return float(-0.5 * d @ (self.S_goal_inv @ d))
