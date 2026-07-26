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

**Results** (2026-07-26, 7 arms x 100 seeds x 300 s, `results/run-100seed-esc-readapt/`;
a 10-seed pilot in `results/rerun-10seed-esc-readapt/` agrees within noise):
noadapt 5.18 falls/bout, grid 1.49, bo 1.59, **esc 1.94**, safegp 0.48, aif 0.41,
oracle 0.00. ESC is *worse* than grid and bo (p=6e-9 / 2.5e-6, Mann-Whitney on per-seed
fall counts) and far worse than safegp/aif (p~1e-26) — model-free on-plant dithering
buys nothing over naive search, the intended contrast. safegp vs aif is a tie (p=0.5).
Two framing points: ESC's `propose` is arithmetic-only (max compute latency 0.01 s vs
1.4-1.8 s for the GP arms), so it *wins* on latency and still loses on falls; and the
AIF trigger detects faster than the CUSUM arms (median 0.88 s vs ~1.27 s).
Within-event re-adaptation fires but stays selective: 203 gait applies over 141 events,
62 within-event re-adapts, concentrated in 20/100 seeds (proxy count — see below).

Proxy for counting gait applies: the npz logs carry no per-step param trace, but the
settle window forces `cusum` to exactly 0 for `readapt_hold` = 10.3 s after every apply,
so maximal zero-runs >= 1000 steps while `state`==1 count applies. It UNDER-counts when a
fall truncates the window. Do not count zero-crossings instead — a CUSUM hits its 0 floor
constantly and that gave a ~100x overcount on the first attempt.

**Comparability trap:** the 100-seed CSVs in `results/` and `results/aif-100-uprighttrig/`
are from 2026-07-19, i.e. BEFORE the async-responder refactor committed in 0af436c
(2026-07-25) touched `continual_driver.py`. noadapt alone moved 14.26 -> 5.4 falls/bout
(and 15 -> 6.4 events/bout, 96% -> 84% per-event fall rate) with no change to that arm's
own code, so the old and new numbers must NOT be pooled or compared. Those runs also
used **300 s** bouts, not the 120 s CLI default — always pass `--duration 300`.

Depends on the architecture in [[async-unified-aif-agent]]; paper side in
[[paper-payload-reframe]].
