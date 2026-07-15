# experiment-payload-adapt

Event-triggered online CPG adaptation where the "terrain change" is a change of
the **robot's own dynamics**: the Laikago walks on a flat plane carrying an 8 kg
payload rigidly attached above its trunk; halfway through the bout the payload
**shifts rearward + laterally** (ramped over 1 s), moving the combined CoM off
the sagittal plane.

Rationale (see `notes/adaptation-challenge-candidates.md`): on geometric /
friction terrain, falls were transition shocks and the steady-state objective
saturated, so the oracle never beat no-adapt. An asymmetric payload is a
*persistent* mismatch — it costs attitude effort on every stride until the gait
is re-tuned — and it loads the attitude-PD channel where the tightened goal
prior showed adaptation has traction. The shift is deliberately ramped so the
discriminating signal is sustained mismatch, not an impulse.

Phases: **1** (t < T/2) payload centered (mild, symmetric); **2** (t ≥ T/2)
payload offset (persistent roll/pitch torque bias). The trigger baseline
(2.4–3.4 s) lies inside phase 1, so the monitor detects the *shift*, not the
payload itself.

Implementation: collision-masked box (pure mass) + `JOINT_FIXED` constraint on
the trunk; the shift ramps the constraint's child pivot. Trigger / squash /
safeguard machinery and the method objects are imported unchanged from
`experiment-flat2slope-adapt/run_experiment.py` (single source of truth).

## Protocol

```bash
# 1. Fit per-phase optima AND run the cross-penalty screen (go/no-go):
python experiment-payload-adapt/fit_payload_oracles.py --trials 60 --seeds 3
#    -> results/payload_optima.json; prints V(centered_opt|shifted) vs
#       V(shifted_opt|shifted). If the gap is small, increase --mass /
#       shift offsets before burning compute on the main comparison.

# 2. Main comparison:
python experiment-payload-adapt/run_experiment.py run \
    --seeds 20 --arms noadapt grid bo marxefe oracle --workers 10

# 3. Aggregate + figures:
python experiment-payload-adapt/analyze.py
```

Key options: `--mass` (kg), `--shift-lat` / `--shift-back` (m), `--duration`
(shift always at duration/2), `--trigger {ce,dt,cusum}`, `--no-attitude-fb`.

## Metrics

Fall counts alone can miss a persistent mismatch (no-adapt may *limp* rather
than fall), so the manifest logs per-phase means of velocity tracking, tip
deviation, actuator mechanical power Σ|τ·q̇| and cost of transport, plus the
trigger's detection latency relative to the shift.

## Regime map (screened 2026-07, incumbent = flat-optimal, feedback ON)

| mass / offset | outcome after mid-run shift |
|---|---|
| 4 kg / 0.10–0.15 m | absorbed by attitude feedback: roll bias −0.5°, no trigger |
| 6 kg / 0.20 m | survives, vx 0.57→0.48, ratio ≈1.1 (below K=2, no trigger) |
| **8 kg / 0.20 m (default)** | survives but **limps** (vx ≈0.27; a screened gait reaches 0.74 under the same load); CE trigger fires 1–4 s post-shift and re-fires |
| 8 kg / 0.225 m | falls 1/3 seeds; trigger fires |
| 8 kg / 0.25 m | incumbent falls 3/3 ~2–5 s post-shift; of 8 LHS gaits only backward-walkers survive — likely unrecoverable (ice-like), avoid as default |
| 10 kg / 0.30 m | falls ~0.7 s after ramp: pure transition shock |

So at the default the discriminating signal is the **phase-2 velocity / J /
power gap**, with falls appearing as the offset grows. Do not fit oracles with
a payload attached pre-shifted at rest: settling a crouched robot under the
full 0.25 m offset tips it statically — the fit script therefore shifts at
t = 1 s while walking.

Variants worth trying if the default is too easy/hard: `--shift-lat 0.225`
(adds falls), heavier payload, or an add-at-T/2 payload instead of a shift
(set phase 1 = no payload).
