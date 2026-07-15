"""GP-safe recovery agent: a high-level agent that DISCOVERS a recovery gait from
experience, with a persistent memory over the control space.

Motivation (see the sweep arc in this folder's README / memory): a per-step MARX
linear model plus isotropic exploration cannot reach the leg-damage recovery gait
without tipping the robot, and the recovery direction lives in parameters
(coupling, b) whose effect only appears post-damage. This agent instead keeps an
EXPLICIT memory over the control space -- a Gaussian process mapping CPG
parameters -> post-damage performance V -- that remembers which parameter choices
FELL (V = V_FALL). Across repeated damage episodes the memory persists, so the
agent avoids the fall regions it has seen and homes in on a survivor. No gait is
ever supplied; the agent builds its own map.

Two design choices the user allowed:
  * reduced control space -- only FREE_DIMS of the 8 CPG params are searched; the
    rest are frozen at the incumbent (the healthy<->damaged optimum barely moves
    them: w_stance / knee_amp are identical, F_FAST nearly so). This shrinks the
    search from 8-D to ~5-D.
  * persistent memory -- the (params, V, fell) archive is saved to / loaded from
    disk, so fall information accumulates across separate runs, not just trials.

Acquisition is SAFE GP-UCB: over a Sobol pool it takes the max upper-confidence
(mu + beta*sigma) among candidates whose lower-confidence (mu - kappa*sigma)
clears a survival threshold SAFE_V; if the safe set is empty (early, uncertain)
it probes the safest frontier (max LCB) to gather safe data without diving into
a predicted fall. This is SafeOpt-style exploration over the agent's own memory.

The agent is deliberately decoupled from the sim: `propose()` returns a full
8-vector to evaluate, `update(full_x, V, fell)` folds the outcome into the GP.
"""

import os
import numpy as np

# 8 CPG params: [coupling, w_swing, w_stance, F_FAST, STOP_GAIN, hip_amp, knee_amp, b]
FREE_DIMS_DEFAULT = [0, 1, 4, 5, 7]   # coupling, w_swing, STOP_GAIN, hip_amp, b
V_FALL = -2.0                         # a fall scores this (matches fit scorer)


class GPSafeRecovery:
    def __init__(self, incumbent, box, free_dims=None, seed=0,
                 n_init=6, safe_V=-0.8, beta=2.5, kappa=1.5, pool=1024,
                 r_fall=0.22, r_rep=0.04, objective="ucb",
                 efe_y_star=1.0, efe_tau2=0.5, efe_adaptive=False,
                 efe_tau2_min=0.1, efe_tau2_max=3.0, archive_path=None):
        self.inc = np.asarray(incumbent, float)
        self.lo = np.asarray(box[0], float)
        self.hi = np.asarray(box[1], float)
        self.fd = list(FREE_DIMS_DEFAULT if free_dims is None else free_dims)
        self.rng = np.random.default_rng(int(seed))
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.safe_V = float(safe_V)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.pool = int(pool)
        self.r_fall = float(r_fall)    # UCB: avoid this radius around remembered falls
        self.r_rep = float(r_rep)      # never re-evaluate within this radius
        # acquisition objective: "ucb" (mu+beta*sigma, hard fall-exclusion) or
        # "efe" (Anil Meera & Kouw, LCSYS 2026 -- eq. 5; safety from the goal prior)
        self.objective = str(objective)
        self.efe_y_star = float(efe_y_star)      # preferred outcome y* (optimistic V)
        self.efe_tau2 = float(efe_tau2)          # preference variance tau^2 (fixed)
        self.efe_adaptive = bool(efe_adaptive)   # curvature-aware tau^2 (eq. 11-12)
        self.efe_tau2_min = float(efe_tau2_min)
        self.efe_tau2_max = float(efe_tau2_max)
        self.d = len(self.fd)
        self.flo = self.lo[self.fd]
        self.fhi = self.hi[self.fd]
        # memory: full 8-vectors, V, fell (parallel lists)
        self.Xfull, self.Y, self.Fell = [], [], []
        self.archive_path = archive_path
        if archive_path and os.path.exists(archive_path):
            self.load(archive_path)

    # ── reduced <-> full ─────────────────────────────────────────────────────
    def _norm(self, full):
        return (np.asarray(full, float)[self.fd] - self.flo) / (self.fhi - self.flo)

    def _expand(self, xr_norm):
        full = self.inc.copy()
        full[self.fd] = self.flo + np.clip(xr_norm, 0.0, 1.0) * (self.fhi - self.flo)
        return full

    # ── GP fit + prediction (botorch SingleTaskGP) ───────────────────────────
    def _fit(self):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        X = np.array([self._norm(x) for x in self.Xfull])
        Y = np.array(self.Y, float)
        Xt = torch.tensor(X, dtype=torch.double)
        Ym, Ys = Y.mean(), max(Y.std(), 1e-6)
        Yt = torch.tensor(((Y - Ym) / Ys)[:, None], dtype=torch.double)
        model = SingleTaskGP(Xt, Yt)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        noise = float(model.likelihood.noise.mean().item()) * Ys ** 2   # sigma_n^2, V units
        return model, float(Ym), float(Ys), max(noise, 1e-9)

    def _mean_std(self, model, Ym, Ys, pool_u):
        import torch
        with torch.no_grad():
            post = model.posterior(torch.tensor(pool_u, dtype=torch.double))
            mu = post.mean.squeeze(-1).numpy() * Ys + Ym          # latent mean, V units
            sig = np.sqrt(post.variance.squeeze(-1).numpy()) * Ys  # latent std,  V units
        return mu, sig

    def _laplacian(self, model, Ys, pool_u, h=0.03):
        """|sum_i d^2 mu/dx_i^2| of the GP posterior mean over the pool (V units),
        by central finite differences in the unit cube (curvature-aware tau^2)."""
        import torch
        with torch.no_grad():
            base = model.posterior(torch.tensor(pool_u, dtype=torch.double)
                                   ).mean.squeeze(-1).numpy()
            lap = np.zeros(len(pool_u))
            for i in range(self.d):
                e = np.zeros(self.d); e[i] = h
                mp = model.posterior(torch.tensor(np.clip(pool_u + e, 0, 1),
                                     dtype=torch.double)).mean.squeeze(-1).numpy()
                mm = model.posterior(torch.tensor(np.clip(pool_u - e, 0, 1),
                                     dtype=torch.double)).mean.squeeze(-1).numpy()
                lap += (mp - 2.0 * base + mm) / (h * h)
        return np.abs(lap) * Ys

    def _sobol(self, n):
        try:
            from scipy.stats.qmc import Sobol
            m = int(np.ceil(np.log2(max(2, n))))
            u = Sobol(d=self.d, seed=int(self.rng.integers(1 << 30))).random_base2(m)
            return u[:n]
        except Exception:
            return self.rng.uniform(0, 1, size=(n, self.d))

    # ── propose / update ─────────────────────────────────────────────────────
    def propose(self):
        """Return a full 8-vector to evaluate next, using the chosen objective.

        objective="ucb": GP-UCB (mu + beta*sigma) with the agent's fall memory
        carved out -- any pool point within r_fall of a REMEMBERED fall is hard-
        excluded (explicit "don't repeat a fall").

        objective="efe": minimize the Expected Free Energy acquisition (Anil Meera
        & Kouw, LCSYS 2026, eq. 5)
            G(x) = (mu-y*)^2/(2 tau^2) + (sig^2+sig_n^2)/(2 tau^2)
                                              - 1/2 ln(1 + sig^2/sig_n^2)
        with a Gaussian preference N(y*, tau^2) over the outcome. Safety is NOT a
        hard radius here -- it emerges from the goal prior: fall regions have low
        mu, far below the optimistic y*, so their pragmatic term inflates G and
        they are avoided. tau^2 is fixed (efe_tau2) or curvature-aware (eq. 11-12).
        Both objectives keep only the repeat guard (never re-evaluate a point)."""
        if len(self.Y) < self.n_init:
            return self._expand(self.rng.uniform(0.0, 1.0, size=self.d)), "init"
        pool = self._sobol(self.pool)
        model, Ym, Ys, noise = self._fit()
        mu, sig = self._mean_std(model, Ym, Ys, pool)
        Xn = np.array([self._norm(x) for x in self.Xfull])

        def _mindist(mask):
            if not mask.any():
                return np.full(len(pool), np.inf)
            return np.linalg.norm(pool[:, None, :] - Xn[None, mask, :], axis=2).min(axis=1)

        d_any = _mindist(np.ones(len(Xn), bool))
        repeat = d_any < self.r_rep

        if self.objective == "efe":
            if self.efe_adaptive:                       # curvature-aware tau^2 (eq. 11-12)
                curv = self._laplacian(model, Ys, pool)
                tau_i2 = 1.0 / (curv + 1.0 / np.maximum(sig ** 2, 1e-9))
                tau2 = self.efe_tau2_min + (self.efe_tau2_max - self.efe_tau2_min) \
                    * tau_i2 / max(np.max(tau_i2), 1e-12)
            else:
                tau2 = self.efe_tau2
            sig_y2 = sig ** 2 + noise
            pragmatic = ((mu - self.efe_y_star) ** 2 + sig_y2) / (2.0 * tau2)
            epistemic = 0.5 * np.log1p(sig ** 2 / noise)
            G = pragmatic - epistemic                   # minimize
            G = np.where(repeat, np.inf, G)
            idx = int(np.argmin(G))
            mode = "efe-adapt" if self.efe_adaptive else "efe"
            return self._expand(pool[idx]), mode

        # ── UCB (default) ────────────────────────────────────────────────────
        ucb = mu + self.beta * sig
        d_fall = _mindist(np.array(self.Fell, bool))
        blocked = (d_fall < self.r_fall) | repeat
        if blocked.all():
            lcb = mu - self.kappa * sig
            return self._expand(pool[int(np.argmax(lcb))]), "frontier"
        idx = int(np.argmax(np.where(blocked, -np.inf, ucb)))
        mode = "explore" if sig[idx] > 0.5 * sig.max() else "refine"
        return self._expand(pool[idx]), mode

    def update(self, full_x, V, fell):
        self.Xfull.append(np.asarray(full_x, float).copy())
        self.Y.append(float(V))
        self.Fell.append(int(bool(fell)))
        if self.archive_path:
            self.save(self.archive_path)

    def best(self):
        i = int(np.argmax(self.Y))
        return self.Xfull[i].copy(), float(self.Y[i])

    def n_safe_seen(self):
        return int(sum(1 for v in self.Y if v > self.safe_V))

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, Xfull=np.array(self.Xfull, float),
                 Y=np.array(self.Y, float), Fell=np.array(self.Fell, int),
                 free_dims=np.array(self.fd, int))

    def load(self, path):
        d = np.load(path, allow_pickle=True)
        self.Xfull = [np.asarray(x, float) for x in d["Xfull"]]
        self.Y = list(np.asarray(d["Y"], float))
        self.Fell = list(np.asarray(d["Fell"], int))
        return len(self.Y)
