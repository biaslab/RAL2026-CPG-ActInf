# experiment-simulation/experiment-payload-adapt

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

The bout is **continual**, not two-phase: the robot walks with the payload
centered until the first event at `first_event_t` (4 s), the payload then shifts
(persistent roll/pitch torque bias) and **stays** shifted until the robot falls;
a fall recentres it and it re-engages after a random `--gap-min`..`--gap-max`
gap. The detector's baseline comes from the settled walking before the first
event, so the monitor detects the *shift*, not the payload itself.

Implementation: collision-masked box (pure mass) + `JOINT_FIXED` constraint on
the trunk; the shift ramps the constraint's child pivot. Trigger / squash /
safeguard machinery and the method objects are imported unchanged from
`methods/{continual_driver,event_responders,responder_worker}.py` (single source
of truth; the precursor `experiment-flat2slope-adapt/` now lives under
`archive/experiments/`).

## Protocol

```bash
# 1. Fit per-phase optima AND run the cross-penalty screen (go/no-go):
python experiment-simulation/experiment-payload-adapt/fit_payload_oracles.py --trials 60 --seeds 3
#    -> results/payload_optima.json; prints V(centered_opt|shifted) vs
#       V(shifted_opt|shifted). If the gap is small, increase --mass /
#       shift offsets before burning compute on the main comparison.

# 2. Main comparison:
python experiment-simulation/experiment-payload-adapt/run_experiment.py \
    --arms noadapt grid bo esc safegp oracle aif \
    --seeds 100 --duration 300 --jobs 20

# 3. Aggregate + figures:
jupyter nbconvert --execute --inplace experiment-simulation/experiment-payload-adapt/analyze.ipynb
```

Payload options: `--mass` (kg), `--shift-lat` / `--shift-back` (m),
`--shift-ramp-t` (s, how abruptly the shift engages).

Key options: `--arms` (any of `noadapt grid bo esc safegp oracle aif`; the
default is `event_responders.ALL_ARMS`, which does **not** include `aif` -- name
it explicitly, as the reported runs do),
`--seeds`, `--duration`, `--jobs` (parallel workers), `--out-dir` (write results
elsewhere so the canonical ones are not overwritten), `--free-dims` (which CPG
dims the arms search), `--sim-speed` (real-time pacing; this is what makes a
slow `propose()` cost simulation steps), and the CUSUM detector knobs
`--detect-kappa` / `--detect-h` / `--detect-tau` / `--no-detector`.

Note `--duration 300`: the CLI default is 120 s and is not what the reported
runs use (see `results/PROVENANCE.md`).

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

## Fall-rate sweep (2026-07, `sweep_fallrate.py`)

Systematic sweep for a HIGH no-adapt fall rate that is still recoverable
(10 seeds/setting, 30 s bouts unless noted; "recoverable" = a BO-fit gait
walks forward under the always-shifted condition):

| setting (mass / lat / back) | fall% | notes |
|---|---|---|
| 8 / 0.20 / 0.20 (old default) | 0% | limps, vx2 ~0.38 |
| **8 / 0.225 / 0.225** | 70% @30 s, **90% @60 s** | falls +5..+30 s post-shift, 0 shocks; **recoverable**: BO gait V=+0.48, vx=0.46, 0 falls |
| 8 / 0.23 / 0.23 | 70% @30 s | no gain over 0.225 |
| 8 / 0.2375 / 0.2375 | 90% | UNRECOVERABLE (BO best V=-0.50, backward only) |
| 8 / 0.25 / 0.25 | 100% | unrecoverable (confirms earlier screen) |
| 9 / 0.20 / 0.20 | 50% | late falls (+13.6 s) |
| 9 / 0.225 / 0.225 | 100% | UNRECOVERABLE (BO best V=-0.59) |
| 10 / 0.20-0.225 | 90-100% | 30% shocks, incl. pre-shift falls: phase 1 unsafe |
| 8 / lat 0.30 / back 0.10 | 100% | fast falls (+2.2 s), unrecoverable (LHS: only backward) |
| 8 / lat 0.10 / back 0.30 | 0% | pure rearward shift is benign (vx2 even improves) |
| 8 / 0.20 / 0.20, up 0.30 | 20% | raised payload only mildly destabilises |

Take-aways: the **lateral offset is the destabilising axis** (sagittal shifts
are absorbed by pitch); the recoverability cliff is razor-thin (0.225
recoverable, 0.2375 not); >=10 kg breaks the safe-phase-1 design. For a
fall-dominated regime run

```bash
python experiment-simulation/experiment-payload-adapt/run_experiment.py run --shift-lat 0.225 --shift-back 0.225 ...
python experiment-simulation/experiment-payload-adapt/fit_payload_oracles.py   # refit optima at the new offsets!
```

(no-adapt then falls 9/10 seeds; the surviving seed limps at vx ~0.07).
Sweep data: `results/sweep_fallrate.csv`, `results/sweep_screen.csv`,
`results/sweep_bo_recover.csv`, `results/sweep_fallrate_B_60s.csv`.
