---
name: backward-gait-ceiling
description: At the 0.215/0.20 m payload offset every arm including the clairvoyant oracle settles into a BACKWARD-walking gait, so falls-avoided is the only honest metric
metadata:
  type: project
---

Measured on the 2026-07-26 100-seed payload run, mean forward velocity in the window
15-90 s after adaptation (before any later fall):

| arm | mean $v_x$ | bouts walking backward |
|---|---|---|
| oracle | -0.216 m/s | 100% |
| aif (ours) | -0.116 m/s | 99% |
| safegp | -0.102 m/s | 95% |
| grid / bo / esc | -0.137 / -0.139 / -0.136 m/s | 86-87% |

Target is +0.5 m/s forward. The clairvoyant oracle being the *most* retrograde is the
load-bearing detail: the pre-fit gait that never falls is itself a backward one, so
within `FREE_DIMS_PAYLOAD` no gait both tolerates the offset and advances. Staying
upright and making progress are in tension at this offset.

**Why:** it invalidates any claim that adaptation preserves locomotion here, and it
means "V ~ 0 / braces against the load" (the earlier wording in root.tex) understated
the cost. The falls comparison stays clean because every arm pays the same price.

**How to apply:** report falls-avoided as the benefit and say plainly that forward
travel is not preserved; do not quote distance-under-fault as a success metric. If a
reviewer asks for adaptation that keeps walking forward, that needs a richer
parameterization, not a better search. Related: [[esc-baseline-and-readapt]],
[[payload-shift-experiment]].
