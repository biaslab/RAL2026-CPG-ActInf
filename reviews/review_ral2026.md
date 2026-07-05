# Peer Review — IEEE Robotics and Automation Letters (RA-L)

**Manuscript:** *Rapid terrain adaptation through continual learning of central pattern generator parameters for quadrupedal robot locomotion*

**Reviewer role:** Critical peer reviewer

**Recommendation: Major Revision** (borderline reject — see rationale)

---

## 1. Summary of the submission

The paper studies online adaptation of central-pattern-generator (CPG) parameters for a
quadruped (Laikago, PyBullet) crossing from flat ground onto a 10° incline. It makes three
claims:

1. The CPG parameter vector that maximizes a stability criterion on flat ground differs from
   the one on a 10° incline, and keeping the flat-optimal set after the transition "roughly
   doubles the probability of a fall."
2. An event-triggered prediction-error monitor (a 0.5 s MARX rollout error against a
   flat-walking baseline) reliably detects the terrain change (~0.33 s after crossing).
3. Among three adaptation mechanisms — Latin-hypercube grid search, GP-UCB Bayesian
   optimization (BO), and a MARX-based active-inference agent minimizing Expected Free Energy
   (MARX-EFE) — only MARX-EFE avoids destabilizing the robot, matching an oracle that switches
   directly to the incline optimum, while grid/BO roughly double the fall rate by probing
   candidate gaits on the physical robot.

The paper is clearly written, the statistical reporting is careful (McNemar, Wilcoxon), and the
limitations section is commendably honest. The central *negative* finding — that trial-based
online black-box optimization is actively harmful on a hazardous transition — is genuine and
worth publishing. However, the *positive* contribution (that the proposed method helps) is not
statistically established by the paper's own numbers, and the motivating premise is weaker than
the abstract states. These issues are substantial and, in my judgment, require major revision.

---

## 2. Major concerns

### M1. The proposed method is not statistically better than doing nothing

This is my central concern. Table I:

| Metric | no adapt | oracle | MARX-EFE |
|---|---|---|---|
| Falls, 20 s [%] | 40 | 47 | **39** |
| Falls, 10 s [%] | 19 | 8 | **10** |
| Upright time [s] | 16.7 | 17.3 | **17.8** |

The paper itself states (Sec. IV-D) that MARX-EFE's early fall rate "is statistically
indistinguishable from the oracle **and from no adaptation**," and over 20 s "again matches no
adaptation (40%)." So the headline result is that the proposed method is *statistically equal to
the do-nothing baseline*. The demonstrated value of MARX-EFE is entirely **not being harmful**,
i.e., not degrading below no-adaptation the way grid/BO do — it is not shown to *improve* on
keeping the incumbent gait.

The abstract and conclusions ("the active inference agent matches an oracle," "the ability to
improve the gait *without* risking it," title: "*Rapid terrain adaptation*") therefore
overclaim. What is actually shown is: (a) online black-box search is destructive, and (b) a
model-based selector avoids that destruction and ends up where no-adaptation already was. The
paper should be reframed around this honestly, and the title's "rapid terrain adaptation" /
"continual learning" framing revisited (this is a single event-triggered re-tune, not continual
learning, and no *improvement* over the incumbent is demonstrated).

### M2. The motivating premise ("terrain-matched parameters matter") is marginal and reverses over the full horizon

The abstract states that keeping flat-optimal parameters "roughly doubles the probability of a
fall." This holds only in the 10 s window (oracle 8% vs no-adapt 19%, McNemar *p* = 0.03 — a
marginal result on a single-scenario dataset) and **reverses over 20 s**, where the oracle
(47%) is *worse* than no-adaptation (40%). In other words, switching to the "correct"
incline-optimal parameters is, over the full observation horizon, no better — numerically worse —
than never switching at all. The paper acknowledges this ("sustained 10° climbing carries a
steady fall hazard for any fixed parameter set"), but the acknowledgment undercuts the entire
motivation: if the terrain-conditioned optimum confers no robust survival advantage, the premise
that "resuming stable locomotion requires online adaptation" is not supported by the evidence.
The claim of "doubling" is a window-selective statement and must be qualified everywhere it
appears (abstract, Sec. II-B, conclusions).

### M3. No direct evidence that MARX-EFE actually adapts the parameters

Given M1 (MARX-EFE ≈ no-adaptation) and the design (control prior centered on current
parameters, ±0.25 trust region, revert-to-best safeguard), the natural hypothesis is that
MARX-EFE largely *stays near the incumbent* — i.e., it is a safe near-no-op rather than an
adapter. The paper provides no figure showing the parameter trajectory MARX-EFE actually
selects, nor whether it converges toward θ\*(10°). Without this, the central mechanistic claim
("selects parameters from a model ... to improve the gait") is unsubstantiated. **Please add a
plot of the selected θ(t) for MARX-EFE (and grid/BO) versus θ\*(10°), aggregated over seeds.**
If MARX-EFE does not measurably move toward the incline optimum, the contribution should be
described as *robust caution* rather than *adaptation*.

### M4. The baselines are weak and under-specified, which inflates the negative result

The comparison rests on grid/BO being "destructive," but:

- The "grid search" baseline (`GridSearchOnline`) walks through a **precomputed 64-point
  Latin-hypercube sequence that is blind to the observed scores** — it is open-loop sampling,
  not a feedback-driven optimizer, and is essentially designed to walk on random gaits. That it
  destabilizes the robot is nearly a foregone conclusion; it is close to a strawman.
- BO gets a single sentence; the GP kernel, length scales, the UCB β schedule, the number of
  random probes, and how the surrogate is warm-started (if at all) are not reported in the
  paper. Given the strong claim that BO "roughly doubles the fall rate," the burden is on the
  authors to show BO was competently configured. As written it is not reproducible from the
  text, and the reader cannot rule out that BO's failure is a tuning artifact.
- Conversely, MARX-EFE receives a full mathematical treatment (Eqs. 9–21). This asymmetry —
  a carefully derived proposed method against thinly described baselines — is a fairness concern
  for a comparative study whose main conclusion is comparative.

At minimum, provide full baseline hyperparameters and a brief justification that they were
tuned, not left at defaults.

### M5. No ablation isolating what the active-inference machinery contributes

The method is presented as active inference / EFE (Eqs. 20–21) with an information-seeking
(epistemic) term. But on a hazardous incline, *exploration is exactly what is penalized* — the
paper's whole thesis is that probing is harmful. It is therefore unclear whether the EFE
formulation, the epistemic term, the goal prior, or the forgetting factor (0.995) matter at all
versus a plain greedy MAP predictive controller (or, indeed, versus a controller that just holds
the incumbent, cf. M1/M3). No ablation is provided. Please justify the active-inference framing
empirically: does the epistemic term help or hurt here, and does MARX-EFE beat a simpler
model-predictive selector?

### M6. Simulation-only, single transition, narrow stochasticity — generalization is not demonstrated

- Evaluation is a **single** flat→10° transition in PyBullet. RA-L strongly values hardware or
  at least broad simulation. The abstract/title promise "terrain adaptation" but only one
  terrain pair is tested; friction changes, declines, steeper/rougher terrain, and repeated
  transitions are all absent.
- The N=100 "seeds" vary *only* by a σ=0.002 rad initial joint perturbation of one identical
  scenario. This is a valid sensitivity measure but a weak basis for the statistical and
  generalization claims: the 100 runs are near-replicates of one condition, not independent
  samples of task difficulty. The p-values quantify sensitivity to micro-perturbations of a
  single scenario, not robustness across terrains or conditions. This should be stated
  explicitly, and broader conditions added.
- Baseline fall rates are very high (39–90%); even the best arm falls 39% within 20 s. The
  intrinsic fragility of the open-loop CPG (acknowledged) is a serious confound: if the
  controller falls ~half the time regardless of parameters, the terrain-adaptation signal is
  small and easily dominated by controller instability. Conclusions may not transfer to a more
  robust base controller.

---

## 3. Moderate concerns

### C1. Incomplete model specification
- Eq. 6 lists θ = [γ, ω_swing, ω_stance, F_fast, K_stop, A_hip, A_knee, b], but the sensory
  feedback gains F_fast and K_stop **do not appear anywhere in the oscillator dynamics
  (Eqs. 1–3)** or the joint mapping (Eqs. 4–5). Where do they enter? The model is not
  self-contained.
- The limit-cycle amplitude `u` appears in Eqs. 1–2 but is neither in θ nor given a value.
- Ajith's TODO note in the source (l. 279) requests pseudocode "for ICRA"; the source uses the
  `ieeeconf` conference class and the note references ICRA, not RA-L. Please confirm the target
  venue and remove leftover author annotations. Pseudocode for the agent would indeed aid
  reproducibility.

### C2. Trigger threshold K=16 generalization
The prediction-error trigger is calibrated (K=16) on held-out seeds — but all seeds are the
*same* flat→10° transition. There is no evidence the trigger generalizes to other terrain
changes, magnitudes, or to false positives from non-terrain disturbances (e.g., pushes,
obstacles). Since the "when to adapt" decision is claimed as a clean, general contribution, at
least one other transition type should stress-test it.

### C3. Representative-seed figures
Figs. 2 and the no-adapt figure show "one representative seed." Given the high variance (roll of
survivors 12–14°, falls in 40–90% of runs), single-seed traces invite cherry-picking concerns.
Prefer median±IQR bands, or state selection criteria.

### C4. Related work is thin and omits the dominant paradigm
Only 11 references, several the authors' own or textbooks. The manuscript does not engage with
learning-based terrain adaptation for quadrupeds (e.g., RL / MPC gait adaptation, rapid motor
adaptation), which is the mainstream approach the reader will compare against. Even without
implementing an RL baseline, the paper must position its contribution relative to that
literature and justify why CPG-parameter adaptation is the right lens. For a journal, the
current coverage is insufficient.

### C5. "Common safety layer" makes the comparison about the safety layer as much as the method
Sec. IV-C states that without the shared trust region + revert-to-best safeguard, "every method,
MARX-EFE included, destabilizes the robot within seconds." This is an important caveat: the
safeguard is load-bearing for all arms. It raises the question of how much of MARX-EFE's success
is the method versus the safeguard doing the work (the safeguard reverts to the best-known =
incumbent = flat params, which is precisely no-adaptation — reinforcing M1/M3). An ablation of
MARX-EFE with/without the safeguard, and how far it can be relaxed for the model-based agent (the
authors' own stated open question), would materially strengthen the paper.

---

## 4. Minor / presentation

- **Abstract**: "roughly doubles the probability of a fall" — qualify as a 10 s-window effect
  (M2). "keeps falls at the level of the best fixed gait" — note that the best fixed gait *is*
  no-adaptation, i.e., the method matches doing nothing (M1).
- **Title**: "continual learning" and "rapid ... adaptation" oversell an event-triggered
  single-shot re-tune with no demonstrated improvement over the incumbent.
- Eq. 7 uses `e` for terrain; the same symbol reads as the natural exponent elsewhere — consider
  a different terrain symbol.
- Fig. 1 caption vs. Table I: define "stability criterion" and V/𝑉̄ consistently; the bottom
  panel's normalization ("shared bounds") should state the bounds.
- Report absolute times for "10 s window" vs. "20 s window" relative to trigger vs. terrain
  crossing consistently; the 0.33 s trigger lag makes the two clocks differ slightly.
- Section IV-D: the phrase "matches an oracle" recurs; given the oracle is itself
  indistinguishable from no-adaptation over 20 s, this is a weak anchor and should be stated as
  such.
- The Appendix and Acknowledgment sections are empty placeholders.

---

## 5. Reproducibility

The MARX-EFE agent is well-specified mathematically. However: baseline (BO/grid)
hyperparameters are absent from the paper (M4); F_fast/K_stop/u are unspecified in the model
(C1); no pseudocode; and the reference optima are computed "beforehand ... with Bayesian
optimization" with no details of that offline procedure (budget, bounds, convergence). A reader
could not reproduce the study from the text alone. Please add a hyperparameter table and, ideally,
release the code (the repository structure suggests this is feasible).

---

## 6. Rationale for recommendation

The paper is honest, readable, and contains a legitimately interesting negative result
(online trial-based optimization is destructive on a hazardous transition). That alone has
value. But as a positive contribution it currently rests on claims its own data do not support:
the proposed method is statistically equal to doing nothing (M1), the motivating "optimum shift"
is marginal and reverses over the horizon (M2), there is no evidence the agent actually adapts
the parameters (M3), the baselines are weak/under-specified (M4), and the active-inference
machinery is not shown to matter (M5) — all on a single simulated transition (M6).

These are addressable, but the required work is substantial: reframed claims, a
parameter-trajectory analysis, ablations (EFE vs. greedy MAP; with/without safeguard), fairer and
fully documented baselines, at least one additional terrain condition to support the
generalization implied by the title, and expanded related work. I therefore recommend **Major
Revision**. I note it is borderline with reject: if a revision cannot demonstrate either a
statistically significant benefit over no-adaptation or genuine adaptation of the parameters
(M1/M3), the core positive contribution would not stand, and the appropriate outcome would be
rejection with encouragement to resubmit the negative-result study in a more focused form.

### Path to acceptance (concrete)
1. Reframe title/abstract/claims around the demonstrated result (harm-avoidance vs. improvement).
2. Add θ(t) trajectory evidence that MARX-EFE adapts toward θ\*(10°) (M3).
3. Show a statistically significant advantage over no-adaptation, or state plainly that none
   exists and re-scope the contribution (M1).
4. Fully document and fairly tune BO/grid; add an EFE-vs-greedy and with/without-safeguard
   ablation (M4, M5, C5).
5. Add ≥1 additional terrain condition (friction and/or decline/steeper) and stress-test the
   trigger (M6, C2).
6. Complete the model specification (F_fast, K_stop, u) and expand related work incl.
   learning-based adaptation (C1, C4).
