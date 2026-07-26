---
name: esc-baseline-and-readapt
description: Uncommitted as of 2026-07-26 — ESC (extremum-seeking) 6th arm and within-event re-adaptation via a settle window; both answer the simulated RA-L review
metadata: 
  node_type: memory
  type: project
  originSessionId: c7e53ce7-85eb-41f9-b57d-5f1001b327a7
  modified: 2026-07-26T10:53:49.895Z
---

Work in the tree but **not yet committed** as of 2026-07-26, both responding to
objections in `reviews/review_2026-07-23_1147.md` (a simulated RA-L reviewer report,
verdict: major revision; three objections — the two experiments didn't run the same
"unified" method, baselines weren't apples-to-apples, and no ablation isolating the
AIF machinery from a plain safety-filtered GP).

**ESC arm.** `ESResponder` in `methods/event_responders.py` adds extremum-seeking
control as a sixth arm (`ALL_ARMS` = noadapt, grid, bo, **esc**, safegp, oracle;
`METHOD_ORDER`/`PALETTE`/etc. in `continual_analysis.py` updated to match). Discrete
ESC after Killingsworth & Krstic (IEEE CSM 26(3):70-79, 2006): one distinct sinusoidal
dither frequency per free dim, demodulate the scalar event score to a per-dim gradient
of J = -V, integrate downhill. Works in normalized [0,1] per-dim coordinates so
amplitude/gain are dimensionless; one propose/update pair per event, so the discrete
step index advances once per event. Two non-obvious details: the washout high-pass is
followed by a **running scale normalization** of |J_hp| — surviving gaits score O(0.1)
while a fall scores O(1), and without normalizing, one gain either crawls or explodes;
and k=0 gives zero dither, so the first proposal is the incumbent, the same anchor
grid/bo use. **Why it's the right baseline:** it's the model-free classical online
tuner, and it has no fall memory, so every proposal *including the exploratory dither*
executes on the already-shifted robot — exactly the destabilizing on-plant sampling
safegp/AIF avoid. That contrast is the point of the comparison.

**Within-event re-adaptation** (`methods/continual_driver_aif.py`). Previously the
trigger fired once per event; now it stays live for the whole event and re-fires
whenever error re-accumulates, so adaptation can recur *without* needing a fall to
reset. Mechanism: after a gait is applied, `agent.S` (the CUSUM) is zeroed and held at
zero for `readapt_hold = cfg.param_ramp * DT + cfg.eval_hold` — only error persisting
past the new gait's ramp-in can re-accumulate, which prevents chattering on the ramp
transient. On a re-fire the *outgoing* gait's outcome is folded back in
(`agent.update(cand, y_obs, False)`) before the new proposal. Bookkeeping split:
`have_gait` (an adapted gait is active) replaced `proposed` in the state log and the
end-of-event scoring; `detect_t` is now set only on the *first* activation, and the
`detect_timeout` liveness override only applies while `detect_t is None`.

Depends on the architecture in [[async-unified-aif-agent]]; paper side in
[[paper-payload-reframe]].
