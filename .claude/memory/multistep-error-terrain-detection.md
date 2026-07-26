---
name: multistep-error-terrain-detection
description: One-step MARX prediction error barely registers a terrain change; the 0.5 s rollout error spikes ~5x and works as a terrain-change detector
metadata: 
  node_type: memory
  type: project
  originSessionId: d3bc3ce8-bfbe-4e8f-865c-7525e674ffdd
---

Found while building the speed-bump demo (`demo-speedbump/`, July 2026): with the
MARX model at dt = 10 ms, the **one-step** innovation ‖y − ŷ‖ is dominated by
persistence and rose only ~1.5–2x when the Laikago walked onto a 10° ramp — not
separable from gait noise (same for the Mahalanobis/surprise version). The
**H = 50-step (0.5 s) rollout error** (predictions issued 0.5 s earlier under
constant controls, compared when reality arrives) spiked ~5x over the flat
baseline and triggered reliably at ramp contact (EMA > baseline mean + 6σ).

**Why:** multi-step rollout compounds model mismatch; one-step prediction can
ride on y_{k} ≈ y_{k-1}.

**How to apply:** for terrain-change detection or adaptation triggers in the
MARX-EFE work, monitor a multi-step rollout error, not the one-step innovation.
Also: `forgetting ≈ 0.995` lets the error re-converge after the change (λ = 1
leaves it elevated). Rollout warmup matters — gate predictions until ~150
updates or the early transient swamps the baseline.
