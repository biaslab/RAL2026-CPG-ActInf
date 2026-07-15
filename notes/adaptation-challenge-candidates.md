# Candidate locomotion challenges where adaptation should actually pay off

*2026-07 — brainstorm after flat→slope, friction, and natural-transect experiments
all showed oracle ≈ no-adapt.*

## Diagnosis: why terrain transitions haven't produced an oracle gap

The experiments so far document a specific failure mode: on geometric/friction
terrain, falls come from the **transition shock itself**, not from sustained
parameter mismatch. The CPG + attitude-PD combination is robust enough that the
steady-state stability objective J saturates over a wide region of the 8-D
parameter space — the flat/grass optimum is "good enough" everywhere we've
tested, so a clairvoyant per-terrain oracle has nothing to gain.

The fix is therefore not *a harder terrain*, but a challenge that

1. changes the robot's dynamics **persistently** (cost accrues every stride,
   not once at a boundary), and
2. moves the optimum along axes the parameterization can actually express
   (global CPG frequencies `w_swing`/`w_stance`, amplitudes, `STOP_GAIN`,
   attitude PD gains).

## Candidates (all PyBullet-native — no simulator switch)

### 1. Payload / center-of-mass change  ← top pick

Mid-run, add 30–60 % trunk mass, optionally offset rearward or laterally
(`changeDynamics` on the trunk, or a fixed-constraint-attached block). Classic
adaptive-control setting: the unloaded-optimal gait is *systematically* wrong
under load — attitude gains undersized, stance frequency and amplitudes
mistuned — and the error accumulates every stride. A lateral/rear CoM offset
loads the attitude-PD channel directly, which is where the goal-prior result
showed adaptation has traction. Easy paper story (load carriage), and
reversible mid-episode (pick up / drop off cargo) so the trigger gets both an
onset and an offset.

→ implemented in `experiment-payload-adapt/` (payload shifts at T/2).

### 2. Viscous resistance — wading through water or mud

Per-step `applyExternalForce` of −c·v on any link below a "surface level",
optionally plus buoyancy. The cheapest route to what deformable terrain was
supposed to provide: a strongly **rate-limiting medium**. Drag punishes fast
swing quadratically, so the optimal `w_swing`/`w_stance` drops substantially —
a large optimum shift on parameters already being adapted, with the penalty
for not adapting paid continuously.

### 3. Actuator degradation / leg damage

Scale down `maxForce` on one leg's motors (or add joint damping) mid-run — the
Cully et al. (Nature 2015) setting, the canonical demonstration that adaptation
beats a fixed policy. **Caveat for this repo**: the 8 CPG parameters are
global/symmetric across legs, so compensation must come indirectly (slower
gait, more stance, higher `STOP_GAIN`, attitude gains). Screen first whether
the parameter space can express a compensating gait at all; if the
damaged-condition oracle is no better than no-adapt, the parameterization —
not the terrain — is the bottleneck.

### 4. Compliant ground via contact parameters

`changeDynamics(ground, -1, contactStiffness=…, contactDamping=…)` gives
penetration-based soft ground — an approximation of mud/foam sinkage with no
simulator switch. Feet sink and lose return energy, so effective leg length and
needed clearance change. Weaker effect than #2, but a one-line variant of the
existing friction-band infrastructure (bands could carry stiffness instead
of μ).

### 5. Battery / torque droop

A slow global decay of motor torque limits over the episode. Uniquely stresses
the *event-trigger* story: a drift rather than a step, so the CUSUM can be
shown accumulating suboptimality evidence. Supporting experiment, not a
headline.

## Methodological guardrails (regardless of candidate)

- **Cross-penalty screen before the full pipeline.** Fit θ\* per condition
  (the `fit_surface_oracles.py` machinery), then evaluate θ\*_A under
  condition B *in steady state* (excluding the transition window). Only commit
  to a condition where the off-diagonal penalty is large. Every terrain that
  failed so far would have failed this screen cheaply.
- **Reconsider the metric.** Fall counts saturate and are transition-dominated.
  Cost of transport / mechanical power and speed-tracking error accumulate
  continuously and expose sustained mismatch that fall counts can't — for
  payload and drag especially, "no-adapt limps along inefficiently" may be the
  real signal even when it doesn't fall.

## Recommended first experiment

Payload with lateral + rearward CoM shift, switching mid-run, scored on falls
**plus** mechanical power / velocity tracking. Hits the attitude-feedback
channel, honest physical story, ~a day of implementation, and the
cross-penalty screen answers within an hour of compute whether the oracle gap
finally exists.
