"""Uniform per-event responders for the continual event-adaptation experiments
(payload-shift, leg-damage). Every responder exposes the SAME two-call interface
that the continual driver (`experiment-*/run_experiment.py`) uses:

    cand, mode = responder.propose()     # full 8-vector CPG params for this event
    responder.update(cand, V, fell)      # fold the event outcome back into memory

Memory persists across events WITHIN a trial, so the feedback-driven responders
(bo, esc, safegp) accumulate a recovery map/estimate over the recurring events.
`noadapt` and `oracle` are memoryless anchors; `grid` ignores feedback.

The arms, matching the experiment design:
  * noadapt -> hold the flat-optimal gait through the event (lower anchor);
  * grid    -> Latin-hypercube, space-filling proposals (ignores feedback);
  * bo      -> GP-UCB on the per-event stability score V;
  * esc     -> discrete extremum-seeking control: a distinct-frequency sinusoidal
               dither per dim, demodulating the scalar score V to estimate the
               cost gradient and integrate the estimate downhill (a model-free
               classical online tuner; Killingsworth & Krstic, IEEE CSM 2006);
  * safegp  -> the safe GP recovery agent (methods.gp_safe_agent), which steers
               its search away from control regions it remembers falling;
  * oracle  -> clairvoyant: jump straight to the pre-fit post-event optimum
               (upper anchor on any parameter switch).

All search-based responders (grid, bo, esc, safegp) act on a REDUCED control
space (`free_dims`); the frozen dims stay at the incumbent. This is the fair
head-to-head: every search arm tunes exactly the dims safegp tunes. Note that
grid, bo, and esc can only score a candidate by TRIALING the (possibly
destabilizing) gait on the already-shifted robot -- esc must even execute its
exploratory dither on the plant -- whereas safegp screens candidates against its
memory first; this contrast is the point of the comparison.
"""

import numpy as np

ALL_ARMS = ["noadapt", "grid", "bo", "esc", "safegp", "oracle"]


class NoAdaptResponder:
    """Lower anchor: keep the incumbent (flat-optimal) gait through every event."""
    mode = "noadapt"

    def __init__(self, incumbent):
        self.inc = np.asarray(incumbent, float)

    def propose(self):
        return self.inc.copy(), self.mode

    def update(self, cand, V, fell):
        pass


class OracleResponder:
    """Upper anchor: clairvoyantly switch to the pre-fit post-event optimum."""
    mode = "oracle"

    def __init__(self, target):
        self.target = np.asarray(target, float)

    def propose(self):
        return self.target.copy(), self.mode

    def update(self, cand, V, fell):
        pass


class GridResponder:
    """Latin-hypercube, space-filling candidate sequence over the reduced dims.
    Ignores outcome feedback (naive search); the first proposal re-tries the
    incumbent (anchor), then space-filling points, cycling if exhausted."""
    mode = "grid"

    def __init__(self, incumbent, box, free_dims, seed, n=64):
        from scipy.stats.qmc import LatinHypercube
        self.inc = np.asarray(incumbent, float)
        self.fd = list(free_dims)
        lo = np.asarray(box[0], float)[self.fd]
        hi = np.asarray(box[1], float)[self.fd]
        lhs = LatinHypercube(d=len(self.fd), seed=int(seed)).random(n=int(n))
        self.seq = lo + lhs * (hi - lo)      # reduced-space points
        self.i = -1

    def _expand(self, reduced):
        full = self.inc.copy()
        full[self.fd] = reduced
        return full

    def propose(self):
        if self.i < 0:                       # anchor: re-try the incumbent first
            self.i = 0
            return self.inc.copy(), self.mode
        cand = self._expand(self.seq[self.i % len(self.seq)])
        self.i += 1
        return cand, self.mode

    def update(self, cand, V, fell):
        pass


class BOResponder:
    """GP-UCB on the per-event stability score V, over the reduced dims. First
    proposal is the incumbent (GP anchor), then `n_random` random probes, then
    decaying-beta UCB proposals fit on the accumulated (gait, V) memory."""
    mode = "bo"

    def __init__(self, incumbent, box, free_dims, seed, n_random=3):
        import torch
        from methods.bo_optimizer import BOOptimizer, BetaSchedule
        self.inc = np.asarray(incumbent, float)
        self.fd = list(free_dims)
        self.lo = np.asarray(box[0], float)[self.fd]
        self.hi = np.asarray(box[1], float)[self.fd]
        self.n_random = int(n_random)
        self.bo = BOOptimizer(
            bounds=torch.tensor(np.vstack([self.lo, self.hi]), dtype=torch.double),
            beta_schedule=BetaSchedule(beta_init=2.0, beta_min=0.5,
                                       n_decay_start=8, gamma=0.8),
            n_init=1 + self.n_random, seed=int(seed))
        self.rng = np.random.default_rng(20_000 + int(seed))
        self.t = 0

    def _expand(self, reduced):
        full = self.inc.copy()
        full[self.fd] = reduced
        return full

    def propose(self):
        if self.t == 0:
            reduced = self.inc[self.fd]
        elif self.t <= self.n_random:
            reduced = self.rng.uniform(self.lo, self.hi)
        else:
            try:
                model = self.bo.fit_model()
                beta = self.bo.beta_schedule(self.t)
                reduced = self.bo.from_unit(self.bo.suggest(model, beta))
            except Exception as e:
                print(f"[bo] GP fit/suggest failed ({e}); random fallback")
                reduced = self.rng.uniform(self.lo, self.hi)
        self.t += 1
        self._last_reduced = np.asarray(reduced, float)
        return self._expand(self._last_reduced), self.mode

    def update(self, cand, V, fell):
        # fold the outcome in at the reduced coordinates that were proposed
        self.bo._append(np.asarray(cand, float)[self.fd], float(V))


class ESResponder:
    """Discrete extremum-seeking control (ESC) over the reduced dims -- the
    model-free classical online tuner (Killingsworth & Krstic, IEEE Control
    Systems Magazine 26(3):70-79, 2006). It holds a parameter estimate, perturbs
    each dim with a distinct-frequency sinusoidal dither, and demodulates the
    scalar event score to estimate the per-dim cost gradient, integrating the
    estimate downhill on the cost J = -V.

    Unlike the model-based agents it has NO model and NO fall memory to screen
    against, so every proposal -- including the exploratory dither -- is executed
    on the already-shifted robot; this is exactly the destabilizing on-plant
    sampling the safe GP / AIF agents avoid. Convergence is known to degrade with
    parameter dimension, so on a larger reduced space this is a deliberately weak
    (but fair) classical baseline.

    Works in normalized [0,1] coordinates per dim so the dither amplitude and
    integrator gain are dimensionless; the frozen dims stay at the incumbent. One
    propose()/update() pair per event, so the discrete step index advances once
    per event outcome.
    """
    mode = "esc"

    def __init__(self, incumbent, box, free_dims, seed, amp=0.15, gain=0.08,
                 wash=0.2):
        self.inc = np.asarray(incumbent, float)
        self.fd = list(free_dims)
        self.lo = np.asarray(box[0], float)[self.fd]
        self.hi = np.asarray(box[1], float)[self.fd]
        span = self.hi - self.lo
        self.span = np.where(span > 0, span, 1.0)      # guard degenerate dims
        d = len(self.fd)
        self.a = float(amp)                            # dither amplitude (unit space)
        self.gain = float(gain)                        # integrator gain
        self.wash = float(wash)                        # washout (high-pass) EMA rate
        # distinct dither frequencies, spread well inside (0, pi) so each dim's
        # perturbation is separable under demodulation.
        self.omega = np.pi * (0.1 + 0.35 * (np.arange(d) + 1.0) / d)
        self.uhat = np.clip((self.inc[self.fd] - self.lo) / self.span, 0.0, 1.0)
        self.k = 0                                     # discrete step (advances per event)
        self.J_dc = None                               # running cost DC for the washout
        self.J_scale = 1.0                             # running scale of the high-passed cost
        self._demod = np.zeros(d)                      # sin(omega*k) of the last proposal

    def _expand(self, u):
        full = self.inc.copy()
        full[self.fd] = self.lo + np.clip(u, 0.0, 1.0) * self.span
        return full

    def propose(self):
        # dither around the current estimate (k=0 gives a zero dither, so the
        # first proposal is the incumbent -- the same anchor grid/bo use).
        self._demod = np.sin(self.omega * self.k)
        u_prop = np.clip(self.uhat + self.a * self._demod, 0.0, 1.0)
        return self._expand(u_prop), self.mode

    def update(self, cand, V, fell):
        J = -float(V)                                  # ESC minimizes cost; higher V better
        if self.J_dc is None:                          # seed the DC on the first sample
            self.J_dc = J
        J_hp = J - self.J_dc                           # washout: strip the slow DC
        self.J_dc += self.wash * J_hp                  # EMA update of the DC estimate
        # scale-normalize the high-passed cost by its running magnitude, so the
        # gain transfers across the (unknown, jumpy) V range -- surviving gaits
        # score ~O(0.1) while a fall scores O(1) -- without this the same gain
        # would crawl on one and explode on the other.
        self.J_scale += self.wash * (abs(J_hp) - self.J_scale)
        z = J_hp / (self.J_scale + 1e-6)
        # demodulate to a per-dim gradient estimate, then integrate downhill.
        grad = (2.0 / max(self.a, 1e-6)) * self._demod * z
        self.uhat = np.clip(self.uhat - self.gain * grad, 0.0, 1.0)
        self.k += 1


class SafeGPResponder:
    """The safe GP recovery agent (methods.gp_safe_agent.GPSafeRecovery)."""
    mode = "safegp"

    def __init__(self, agent):
        self.agent = agent

    def propose(self):
        cand, mode = self.agent.propose()
        return np.asarray(cand, float), mode

    def update(self, cand, V, fell):
        self.agent.update(np.asarray(cand, float), float(V), bool(fell))


def make_responder(name, incumbent, box, free_dims, oracle_target, seed,
                   *, safegp_kwargs=None, grid_n=64, bo_n_random=3):
    """Build one of the five responders. `safegp_kwargs` (dict) is forwarded to
    GPSafeRecovery; the incumbent is pre-seeded there as a known post-event fall
    by the driver, not here."""
    if name == "noadapt":
        return NoAdaptResponder(incumbent)
    if name == "oracle":
        if oracle_target is None:
            raise SystemExit("oracle arm needs a pre-fit optimum (run fit_*_oracles.py)")
        return OracleResponder(oracle_target)
    if name == "grid":
        return GridResponder(incumbent, box, free_dims, seed, n=grid_n)
    if name == "bo":
        return BOResponder(incumbent, box, free_dims, seed, n_random=bo_n_random)
    if name == "esc":
        return ESResponder(incumbent, box, free_dims, seed)
    if name == "safegp":
        from methods import gp_safe_agent as gp
        kw = dict(safegp_kwargs or {})
        agent = gp.GPSafeRecovery(incumbent, box, free_dims=free_dims, seed=seed,
                                  archive_path=None, **kw)
        return SafeGPResponder(agent)
    raise ValueError(f"unknown arm {name!r}; choose from {ALL_ARMS}")
