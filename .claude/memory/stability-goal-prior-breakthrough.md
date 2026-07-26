---
name: stability-goal-prior-breakthrough
description: Stability-aligned goal prior makes MARX-EFE genuinely adapt — first arm to beat no-adaptation; epistemic term matters once the goal is stability not velocity
metadata: 
  node_type: memory
  type: project
  originSessionId: 5f16b388-ca3e-4120-94fd-036b235daf4c
---

Fixing the MARX-EFE goal prior to reward attitude (stability) instead of only
velocity turned the negative result positive (flat→10°, 100 seeds, reachable
trust 1.0, run 2026-07-06). The agent's goal prior had pitch/roll targets 0 but
std 45° (~4.5× too loose vs ψ0=10°); tightening attitude std to 8° = "marxefe_stable".

Falls over 20 s (early 10 s): no-adapt 40% (19%), oracle 47% (8%),
marxefe (velocity goal) 44% (17%), **marxefe_stable 27% (7%)**,
marxefe_stable_greedy (epistemic OFF) 42% (17%).

- **M1 SOLVED:** marxefe_stable is the FIRST arm to statistically beat
  no-adaptation — 27 vs 40% (McNemar p=0.041); early 7 vs 19% (p=0.008, better
  than the oracle's 8%). Also beats old marxefe (p=0.005) and the oracle (p=0.006).
- **M5 SOLVED:** the epistemic (information-seeking) term now MATTERS — stable
  27% (epistemic on) vs stable_greedy 42% (off). Under the old velocity goal it
  did not (greedy≈marxefe). So active inference contributes once the goal is
  meaningful. (W. Kouw's hypothesis was right.)
- **M3 nuance:** it adapts by SMALL, attitude-directed corrections (mean |Δθ|≈
  0.019 normalized, mostly w_stance/F_fast/b), NOT by reaching the distant BO
  oracle θ*(10°) (gap-closed-toward-oracle stays 0). A nearby model-guided
  correction stabilizes; the far BO optimum is one of many stable gaits and need
  not be the target. The gap-closed-toward-θ* metric understates this — falls are
  the real evidence. Note stable MOVES LESS than velocity-marxefe (0.019 vs
  0.032) yet falls far less: smaller, smarter, attitude-aware moves.

**Why:** flips the paper from "robust caution, no improvement" (adverse, M1/M3/M5)
to a genuine positive contribution — stability-aligned active inference makes
small model-guided corrections that significantly cut falls vs no-adaptation, and
the epistemic drive is load-bearing. Supersedes the pessimistic reading in
[[track1-marxefe-no-adaptation]] (that was under the mis-specified velocity goal).

**IMPORTANT TEMPERING (repeated transitions, run 2026-07-06, trust 1.0):** the
single-transition win does NOT generalize. On the 2-ramp staircase:
no-adapt 26%, oracle 11%, marxefe 23%, marxefe_stable 24% (vs no-adapt p=0.83,
NS), marxefe_stable_greedy 21%. So marxefe_stable ≈ no-adapt ≈ velocity-marxefe;
the epistemic benefit also did not carry over (stable 24 vs greedy 21). All
MARX-EFE variants stay 21–24%, far from the oracle's 11% (p=0.007) — the durable
adaptation benefit remains UNCAPTURED on repeated transitions. (Baseline noadapt
is only 26% here vs 40% on the single sustained incline because the flat landings
let the robot recover.) Net: the stability fix helps on a SUSTAINED single
incline but is NOT yet a robust adapter across scenarios. Do not overclaim.

**GOAL-PRIOR SWEEP RESULT (2026-07-06, repeated terrain, trust 1.0, 100 seeds):**
attitude-std {4,8,16}° × control-prior-scale {0.15,0.5}. NO config beats no-adapt
(26%): best is att=4°/tight-prior at 18% (p=0.115, NS); att=16°/tight 20%
(p=0.29); att=8° 24%. gap-closed ≈ 0 for ALL configs (never moves toward
θ*(10°)). Loosening the control prior HURTS (att=4°: 18%→29% at c=0.5) — it
causes wandering, not adaptation. Conclusion: the limit is STRUCTURAL, not
prior-tuning — the linear AR model can't predict which distant params are stable,
so the agent makes small local corrections (caution) but cannot navigate to the
terrain optimum; a better prior does not rescue generalization.

**REDUCED ACTION SPACE (2026-07-06, single 10°, 50 seeds, trust 1.0, only the 4
dims that differ flat vs sloped are free: gamma, w_swing, K_stop, b):** shrinking
the action space does NOT unlock adaptation. no-adapt 42%, oracle 44%, grid 62%
(p=0.04 WORSE), bo 74% (p<0.001 WORSE), marxefe 44%, marxefe_stable 30% (p=0.07).
Black-box search is STILL destructive in 4-D — probing candidate gaits on the
incline tips the robot regardless of dimensionality (rules out "they just need a
smaller space"). marxefe_stable helps the same modest amount as in full space and
STILL has gap-closed=0 (doesn't reach the oracle even 4 reachable dims away). So
the barrier is STRUCTURAL, not dimensional. CLI: --reduced flag (freezes dims
where |flat-sloped|<1e-6), masking in Safeguard._mask.

**How to apply / next:** the fix is promising but scenario-specific, and neither
prior tuning (sweep) nor action-space reduction generalizes it. Structural fix
needed for robust adaptation. Options:
(1) diagnose why it doesn't capture the oracle on repeated (per-ramp re-adapt?
the small corrections aren't right for alternating ramp/flat); (2) goal-prior
sweep over attitude std × control_prior_scale for robustness; (3) frame honestly:
adaptation is necessary+beneficial (oracle), model-based caution avoids black-box
destruction, stability-aligned goal gives a significant gain on the sustained
transition, and robust adaptation across repeated transitions is the open problem.
Code: GOAL_STD class attr + MarxEFEStable/StableGreedy arms in
experiment-eventtrigger/run_experiment.py. See [[repeated-transitions-findings]].
