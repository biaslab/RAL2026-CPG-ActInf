---
name: track1-marxefe-no-adaptation
description: "Track-1 RA-L revision result — MARX-EFE does not adapt θ; only the incumbent-prior+safeguard matter, active-inference terms are inert"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5f16b388-ca3e-4120-94fd-036b235daf4c
---

Track-1 revision experiment (100 seeds, flat→10°, event-trigger harness,
`experiment-eventtrigger/`, run 2026-07-04) reproduced the paper's Table I
(noadapt 40 / oracle 47 / grid 83 / bo 90 / marxefe 39% falls) and added θ(t)
logging + ablations. Findings:

- **M3 — MARX-EFE does not adapt.** Its median selected θ never leaves the flat
  incumbent (γ stays 4, never toward oracle's 12; b stays 10, never toward 1.69).
  Gap-closed toward θ*(10°) ≈ 0.00. It is robust caution, not adaptation.
- **M5 — the active-inference machinery is inert here.** `marxefe_greedy`
  (epistemic term dropped → greedy-MAP) 36% falls and `marxefe_noforget`
  (forgetting=1.0) 35% are statistically identical to full MARX-EFE (39%,
  p≈0.7/0.6 vs marxefe). EFE/epistemic/forgetting are not doing the work.
- **The real mechanism is the incumbent-centered control prior + revert-to-best
  safeguard.** `marxefe_midprior` (prior at mid-bounds) → 99% falls (worse than
  no-adapt). The safeguard reverts every arm to the incumbent by construction,
  so no arm CAN adapt while it is on.

**CONFOUND (found 2026-07-05, W. Kouw's check):** the ±0.25 trust region does
NOT contain θ*(10°). Only 30% of the flat→oracle shift is reachable; the two
most-shifted params are outside — γ (flat 4 → oracle 12) needs trust_radius 1.0,
b (flat 10 → oracle 1.69) needs 0.84. So the "no adaptation / gap-closed=0"
result was measured against a target ~70% unreachable by construction (incl. the
safeguard-off adaptation test, which also used trust 0.25). The oracle sits in a
CORNER (γ at max, b near min), so containing it ≈ full bounds. Proper adaptation
test = re-run with trust_radius 1.0 (reachable). Repeated-transition experiment
is being run at BOTH trust 0.25 (comparable) and 1.0 (reachable).

**Why:** the paper's positive claim ("adapts toward the terrain optimum via
active inference") is not supported by its own data — this is the reviewer's
M1/M3/M5 — BUT the trust-region confound must be cleared (reachable re-run)
before concluding no-adaptation is intrinsic vs a box artifact. Note MARX-EFE's
goal prior is velocity-tracking, not the stability criterion θ* optimizes (M4
objective mismatch), so it may not seek θ*(10°) even when reachable.

**How to apply:** for the revision, reframe away from adaptation + active
inference toward robust model-based caution UNLESS the safeguard-off test shows
otherwise. The decisive follow-up is `marxefe_nosafeguard` (safeguard disabled):
if its θ(t) then moves toward θ*(10°) while staying upright, genuine adaptation
is salvageable; else robust-caution is confirmed. See [[eventtrigger-experiment-findings]].
Code: ablation arms + θ logging + epistemic flag are in
`experiment-eventtrigger/run_experiment.py` and `methods/marxefe_optimizer.py`
(minimizeEFE epistemic kwarg). Full 8-arm CSV backed up in
`results/track1_flat2slope10/`.
