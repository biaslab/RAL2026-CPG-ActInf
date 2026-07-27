---
name: async-unified-aif-agent
description: "2026-07-25 commit 0af436c — out-of-process responders (real propose latency), UnifiedAIFAgent (MARX trigger + reactive GP), and the per-leg damage experiment"
metadata: 
  node_type: memory
  type: project
  originSessionId: c7e53ce7-85eb-41f9-b57d-5f1001b327a7
  modified: 2026-07-26T10:53:35.546Z
---

Commit `0af436c` (2026-07-25, "Added asynch experiments, unified AIF agent, updated
paper") restructured the continual experiments around asynchronous adaptation and a
single active-inference agent.

**Asynchronous responders.** `methods/responder_worker.py` runs the adaptation method
in a *separate process*; the physics loop posts propose-requests and polls a result
queue without blocking. The point is that the wall-clock cost of `propose()` (GP fit
+ acquisition) now translates into elapsed sim steps before the new gait is applied —
i.e. the experiments measure real on-robot adaptation latency instead of pretending
adaptation is instantaneous. Nothing touching PyBullet crosses the process boundary;
the responder is rebuilt inside the worker from a picklable `ResponderSpec`, and
proposals are tagged with `event_id` so stale ones (event already ended) are dropped.

**UnifiedAIFAgent** (`methods/aif_recovery.py`, driven by
`methods/continual_driver_aif.py`). One Gaussian goal prior over
y = [vx, vy, pitch, roll] ties together three parts: a MatrixNormal-Wishart MARX AR
belief updated every sim step at 100 Hz (reusing `marxefe_optimizer.MARXAgent`
*unmodified*, joint angles as exogenous input); a slow event-triggered GP from CPG
params to outputs that proposes a recovery gait by minimizing EFE; and a trigger =
cross-entropy from the MARX one-step predictive to the goal prior, accumulated in a
CUSUM. Key distinction from `MARXAgent`: here the MARX belief only drives the
*trigger* — its CasADi EFE solver is never invoked (only `update` /
`posterior_predictive` / `crossentropy`); the GP selects the gait.

**Damage experiment** (`experiment-simulation/experiment-damage-adapt/`). RR hip+knee maxForce ramps
60→20 Nm, persistent, never auto-healed — only a fall reverts it (stand up, heal,
re-engage after a random 2-8 s gap). The load-bearing design choice: control is
*per-leg* hip amplitude (`PerLegCPG`, 11-D vector, `FREE_DIMS_DAMAGE` = the 4 per-leg
amplitudes). With a single global hip amplitude there is no regime where no-adapt
fails *and* a recovery exists — the asymmetry is irreducible. Per-leg control opens a
broad findable recovery while the symmetric incumbent still falls 4/4.

Mirrors the payload experiment's structure — see [[continual-payload-gpsafe]] and
[[consolidated-continual-experiments]]. Follow-on work in [[esc-baseline-and-readapt]].
