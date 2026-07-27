# Informing the prediction-error trigger from an optimality argument

**Context.** The event trigger currently fires when the MARX-EFE goal
cross-entropy (EMA-smoothed, normalised to a flat baseline) exceeds an empirical
threshold `K ≈ 2`. This note works out a principled alternative: derive the
threshold from a first-/second-order optimality condition on the CPG parameters,
by reading the trigger statistic as a test of whether the current controls are
still at the minimum of the MARX-EFE control objective.

## Setup

MARX-EFE's per-step control objective (expected free energy) has three terms
(`methods/marxefe_optimizer.py`, `EFE` / `_build_efe_solver`):

$$G(u) \;=\; \underbrace{\text{mutualinfo}(x)}_{\text{info gain } I(y;\theta\mid u)}
\;+\; \underbrace{\text{crossentropy}(x)}_{\text{goal / distance-to-goal}}
\;+\; \underbrace{\tfrac12 (u-\mu_0)^\top \Upsilon (u-\mu_0)}_{\text{control cost }=\,-\log p(u)} .$$

Proposal:

1. **Turn the control cost off** (drop the last term).
2. Assume the **information gain is minimal** — flat in $u$. Its only
   $u$-dependence is $-D_y\log(\text{scale})$ with $\text{scale}=1+x^\top\Lambda^{-1}x$;
   once the model is identified ($\Lambda$ large after warm-up, forgetting
   bounded) this is negligible.

Then the signal to trigger on is just the goal cross-entropy $\approx G(u)$, and
triggering becomes a test of whether the current controls $u_{\rm inc}$ (the
CPG parameters) are still at the minimum of $G$.

## What the signal reduces to

Because the predictive covariance is $\Sigma_t = \frac{\text{scale}}{\nu-D_y}\,\Omega$,
the cross-entropy (marxefe_optimizer.py:237) decomposes as

$$G(u) \;=\; \underbrace{\tfrac12\,\mathrm{tr}\!\big(S_*^{-1}\Sigma_t\big)}_{\text{irreducible (variance floor)}}
\;+\; \underbrace{\tfrac12\,(\mu-m_*)^\top S_*^{-1}(\mu-m_*)}_{\text{reducible (mean deviation)}},
\qquad \mu = M^\top x,$$

where $m_*, S_*$ are the goal-prior mean/covariance. Only the second term tracks
"am I predicted to be off-goal." The first is a floor set by predictive
uncertainty that re-tuning $u$ cannot null.

> Note: if $\Sigma_t \approx S_*$ the floor is $D_y/2 = 2$ (for $D_y=4$). That is
> roughly why the empirical "ratio > 2" works — it detects the mean-deviation
> term rising to about the size of the variance floor.

## The subtlety: value ≠ "controls no longer optimal"

"Is $u$ still at the minimum" is a **first-order** statement, and the CE *value*
is not quite it. The incumbent minimises $G$ iff $\nabla_u G(u_{\rm inc})=0$.
Since $\mu = M^\top x$ is linear in $u$ through the input block
$B := \partial\mu/\partial u$ (the learned control→output gain, a slice of
$M^\top$),

$$g \;=\; \nabla_u G \;=\; B^\top S_*^{-1}(\mu-m_*), \qquad
H \;=\; \nabla_u^2 G \;\approx\; B^\top S_*^{-1} B .$$

The gradient is the goal error **pulled back through the control gain**. So CE
can rise in a direction the CPG parameters cannot affect (terrain error that $B$
projects out), leaving the incumbent *still first-order optimal* despite a higher
value. Triggering on the value conflates "off-goal and fixable" with "off-goal
and stuck"; the gradient separates them.

## A first-/second-order trigger statistic

The affine-invariant "distance from optimality" is the **Newton decrement**:

$$\lambda^2 \;=\; g^\top H^{-1} g
\;=\; (\mu-m_*)^\top S_*^{-1} B \,(B^\top S_*^{-1} B)^{-1} B^\top S_*^{-1} (\mu-m_*)
\;=\; \big\lVert \Pi_{\mathcal R(B)}(\mu-m_*) \big\rVert^2_{S_*^{-1}} .$$

Two consequences:

1. It is exactly the **controllable projection** of the goal error — the part
   re-tuning $u$ can remove. This is the "is $u$ still optimal" test done right.
2. Because the MARX surrogate is **linear**, $G$ is exactly convex-quadratic in
   $u$, so $\tfrac12\lambda^2$ is not a second-order approximation — it is the
   **exact suboptimality gap** $G(u_{\rm inc}) - \min_u G$. Triggering on
   $\lambda^2$ triggers on how much CE a full re-optimisation would recover.

## Setting the threshold from principle

**(a) Statistical / false-alarm (χ²).** Under the null "incumbent still optimal;
deviations are just predictive noise $\mu-m_*\sim\mathcal N(0,\Sigma_t)$,"
$\lambda^2$ is a quadratic form of a Gaussian → (generalised) $\chi^2$ with
$r=\mathrm{rank}(B)\le\min(D_u,D_y)$ degrees of freedom. Choose

$$K \;=\; \chi^2_r(1-\alpha)$$

for a target false-positive rate $\alpha$. With $\Sigma_t\approx S_*$ it is a
clean $\chi^2_r$; otherwise whiten by $S_*^{-1/2}\Sigma_t S_*^{-1/2}$
(Satterthwaite), or estimate the single scalar scale from the flat prefix — which
recovers the current baseline calibration but now with a *derived shape and dof*,
not a hand-set level.

**(b) Decision-theoretic (the elegant reading).** The control cost was removed
from the *signal*; put it back as the *threshold*. Adapt only if the predicted CE
gain beats the control-prior price of the Newton step $\Delta u = -H^{-1}g$:

$$\tfrac12\lambda^2 \;>\; \tfrac12\,\Delta u^\top \Upsilon\, \Delta u .$$

i.e. trigger iff the *full* EFE (goal term minus the control cost set aside) would
actually decrease. The control cost's role flips from noise-to-remove to the
natural regulariser deciding *when adaptation pays off* — a cleaner story than a
tuned $K$.

## Caveats

- **The epistemic term is not exactly zero.** Its $u$-dependence is
  $-D_y\log(\text{scale})$, negligible only once identified ($\Lambda$ large).
  Good in the monitoring regime (post-warm-up, steady gait); worth checking
  rather than assuming.
- **$B$ is the multi-step feedthrough**, not one tap — with `delay_inp > 0` the
  control acts over several steps. The `minimizeEFE` graph already encodes
  $\partial(\text{rollout})/\partial u$; the cleanest implementation evaluates
  that objective's gradient and Hessian at $u = u_{\rm inc}$ via CasADi autodiff
  each monitoring step (exact, cheap), rather than hand-forming $B$.
- **The $\chi^2$ scale** depends on the null covariance choice ($\Sigma_t$ vs the
  innovation covariance); that is the one thing still worth calibrating
  empirically.

## Recommendation

$\lambda^2$ (Newton decrement, χ²- or cost-thresholded) is a better trigger than
the CE ratio: same agent, no separate baseline window, a threshold with units and
a false-alarm interpretation, and it fires on *fixable* suboptimality rather than
raw badness. The clean caveat: at ≥15° the fixable part may be small (no stable
gait exists — the error lies largely in the uncontrollable subspace), which
$\lambda^2$ correctly reports as "don't bother," whereas the CE ratio screams.

**Next step (proposed).** Prototype an alternative `TriggerMonitor` that computes
$g, H, \lambda^2$ each step by autodiffing the EFE objective at the incumbent,
logs it alongside the current CE ratio, and compares when each fires on the
existing `archive/experiments/experiment-flat2slope-adapt` runs — to check empirically whether the
χ² threshold and the controllable-projection story hold up on the 10° vs 15° data.
