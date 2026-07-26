---
name: repeated-transitions-findings
description: "Repeated 10° transition experiment — oracle beats no-adapt (adaptation matters), MARX-EFE stays safe but does not adapt even with reachable oracle"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5f16b388-ca3e-4120-94fd-036b235daf4c
---

Repeated-transition experiment (`experiment-eventtrigger`, terrain "repeated" =
2× +10° ramps + flat landings, trigger re-arms per ramp, 100 seeds, run
2026-07-05). Falls over 20 s:

| arm     | trust 0.25 | trust 1.0 (oracle reachable) |
|---------|-----------|------------------------------|
| noadapt | 26%       | 26%                          |
| oracle  | 11%       | 11%                          |
| grid    | 38%       | 87%                          |
| bo      | 41%       | 92%                          |
| marxefe | 22%       | 23%                          |
| greedy  | 21%       | 20%                          |

Trigger re-arming works: ~3.7–4.2 fires/seed (detects each ramp; C2 satisfied),
K=10 (0 FP / 0 miss in calibration).

Key findings:
1. **Adaptation MATTERS here** (unlike single 10°, where oracle≈no-adapt over
   20 s): oracle 11% << no-adapt 26%. On repeated/sustained transitions the
   terrain-matched optimum gives a real, durable survival benefit. Strengthens
   the M2 premise.
2. **MARX-EFE does NOT adapt — confound cleared.** Even at trust_radius 1.0
   (oracle γ=12/b=1.69 fully reachable), MARX-EFE gap-closed toward θ*(10°) =
   0.00; it holds the incumbent. So the no-adaptation result is INTRINSIC, not
   a trust-region artifact. It stays safe (22–23% ≈ no-adapt) but leaves the
   oracle's benefit (11%) unrealized. greedy ≈ marxefe again (epistemic inert).
3. **Black-box search destructive, and trust-region-sensitive:** grid/bo 38/41%
   at trust 0.25, catastrophic 87/92% at full bounds — the safeguard/trust
   region is load-bearing for THEM. MARX-EFE barely changes (22→23%) because it
   stays near the incumbent regardless of box size.

**Why:** gives an honest, stronger narrative — (a) adaptation genuinely matters
on repeated transitions, (b) online black-box search is destructive and needs a
tight box to not be catastrophic, (c) model-based caution is safe but is NOT yet
a real adapter. The unrealized gap marxefe(22%)→oracle(11%) is the opening for a
true adapter. Likely fix = align the agent's goal with the stability criterion
θ* optimizes, not velocity tracking (M4 objective mismatch).
See [[track1-marxefe-no-adaptation]], [[cpg-terrain-feasibility-envelope]].
Figures: results/*_repeated.png and *_repeated_wide.png.
