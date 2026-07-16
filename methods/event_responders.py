"""Uniform per-event responders for the continual event-adaptation experiments
(payload-shift, leg-damage). Every responder exposes the SAME two-call interface
that the continual driver (`experiment-*/run_experiment.py`) uses:

    cand, mode = responder.propose()     # full 8-vector CPG params for this event
    responder.update(cand, V, fell)      # fold the event outcome back into memory

Memory persists across events WITHIN a trial, so the search-based responders
(grid, bo, safegp) accumulate a recovery map over the recurring events. `noadapt`
and `oracle` are memoryless anchors.

The five arms, matching the experiment design:
  * noadapt -> hold the flat-optimal gait through the event (lower anchor);
  * grid    -> Latin-hypercube, space-filling proposals (ignores feedback);
  * bo      -> GP-UCB on the per-event stability score V;
  * safegp  -> the safe GP recovery agent (methods.gp_safe_agent), which steers
               its search away from control regions it remembers falling;
  * oracle  -> clairvoyant: jump straight to the pre-fit post-event optimum
               (upper anchor on any parameter switch).

All search-based responders (grid, bo, safegp) act on a REDUCED control space
(`free_dims`); the frozen dims stay at the incumbent. This is the fair head-to-
head: grid and bo search exactly the dims safegp searches.
"""

import numpy as np

ALL_ARMS = ["noadapt", "grid", "bo", "safegp", "oracle"]


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
    if name == "safegp":
        from methods import gp_safe_agent as gp
        kw = dict(safegp_kwargs or {})
        agent = gp.GPSafeRecovery(incumbent, box, free_dims=free_dims, seed=seed,
                                  archive_path=None, **kw)
        return SafeGPResponder(agent)
    raise ValueError(f"unknown arm {name!r}; choose from {ALL_ARMS}")
