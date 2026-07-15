# experiment-damage-adapt

Event-triggered online CPG adaptation where the change is a **partial actuator
failure in one leg**: the Laikago walks on a flat plane and halfway through the
bout the hip + knee motors of a single **hind** leg lose most of their torque
budget — their `POSITION_CONTROL` maxForce is ramped down from `healthy_force`
(60 Nm) to `damage_force` (22 Nm) over 1 s. This is the Cully et al. (Nature
2015) leg-damage setting, the canonical demonstration that adaptation beats a
fixed policy.

Rationale (see `notes/adaptation-challenge-candidates.md` #3): on geometric /
friction terrain, falls were transition shocks and the steady-state objective
saturated, so the oracle never beat no-adapt. A weakened leg is a *persistent*
mismatch — the motor under-tracks its CPG targets on every stride, dragging and
destabilising the gait until it is re-tuned — so suboptimality accumulates
instead of being a one-off shock. The droop is deliberately ramped so the
discriminating signal is sustained mismatch, not an impulse.

**Caveat / the thing this experiment tests.** The 8 CPG parameters are
**global / symmetric** across legs, so the controller cannot command more torque
to the weak leg. Compensation must come indirectly — a slower gait
(`w_swing`/`w_stance` down, so the weak motor has more time to reach its target
under reduced torque), smaller amplitudes (`hip_amp`/`knee_amp` down, less torque
demanded), higher `STOP_GAIN`, re-tuned attitude gains. Whether the
parameterisation can express a compensating gait **at all** is exactly what the
cross-penalty screen answers. If the damaged oracle is no better than no-adapt,
the parameterisation — not the damage — is the bottleneck.

Phases: **1** (t < T/2) all legs healthy; **2** (t ≥ T/2) one leg's hip+knee
weakened (persistent drag + roll/pitch bias). The trigger baseline (2.4–3.4 s)
lies inside phase 1, so the monitor detects the *damage*, not the healthy gait.

Implementation: the damaged leg's hip+knee `setJointMotorControl2` calls carry
an explicit `force=` that ramps `healthy_force → damage_force`; healthy legs use
the default (uncapped) force, and the abduction hold (500 N) is left intact — the
damage models the sagittal propulsion actuators. Trigger / squash / safeguard
machinery and the method objects are imported unchanged from
`experiment-flat2slope-adapt/run_experiment.py` (single source of truth).

## Protocol

```bash
# 1. Fit per-phase optima AND run the cross-penalty screen (go/no-go):
python experiment-damage-adapt/fit_damage_oracles.py --trials 60 --seeds 3
#    -> results/damage_optima.json; prints V(incumbent|damaged) vs
#       V(damaged_opt|damaged). If the gap is small the global params can't
#       compensate -- lower --damage-force before the main comparison.

# 2. Main comparison:
python experiment-damage-adapt/run_experiment.py run \
    --seeds 20 --arms noadapt grid bo marxefe oracle --workers 10

# 3. Aggregate + figures:
python experiment-damage-adapt/analyze.py
```

Key options: `--leg {FL,FR,RL,RR}` (default `RR`; use a **hind** leg),
`--healthy-force` / `--damage-force` (Nm), `--duration` (damage always at
duration/2), `--trigger {ce,dt,cusum}`, `--no-attitude-fb`.

## Metrics

Fall counts alone can miss a persistent mismatch (no-adapt may *limp* rather than
fall), so the manifest logs per-phase means of velocity tracking, tip deviation,
actuator mechanical power Σ|τ·q̇| and cost of transport, plus the trigger's
detection latency relative to the damage onset. (`t_shift` in the NPZ / metrics
is the damage-onset time, kept named `t_shift` for tooling shared with the
payload experiment.)

## Regime map (screened 2026-07, incumbent = flat-optimal, feedback ON)

Damaging **one leg's hip+knee** maxForce; healthy walking needs ≪ 60 Nm, so
60 Nm ≈ uncapped/default. Peak effect measured over 20 s bouts, 3 seeds, damage
ramped in at t = 1 s while walking; "recoverable" = a re-tuned gait (low
amplitude / slower) walks forward under the always-damaged condition.

| damaged leg | force [Nm] | incumbent outcome | recoverable? |
|---|---|---|---|
| **front (FR/FL)** | 18–32 | absorbed: vx stays ≈0.54–0.58, no fall — a front leg is support, not propulsion | n/a — no gap |
| hind (RR/RL) | 30 | limps: vx 0.59→0.52, no fall | mild gap |
| hind (RR/RL) | 25 | limps: vx →0.25, no fall over 20 s | low-amp gait → vx 0.33 |
| **hind (RR/RL) 22 (default)** | 22 | **falls ~2/3 @20 s (all 3/3 @60 s)**, vx 0.15 | **recoverable**: low-amp gait walks, vx 0.29, 0 falls |
| hind (RR/RL) | 18 | falls 3/3 | low-amp still walks (vx 0.19) but margin thin |
| hind (RR/RL) | ≤15 | falls immediately (transition shock) | unrecoverable — avoid |

Take-aways: the damaged leg must be a **hind** (propulsion) leg — front-leg
damage is trivially absorbed and produces no oracle gap; the recoverable /
fall-dominated sweet spot is ~22 Nm, and below ~18 Nm the fall becomes a pure
transition shock. Compensation is expressed as a **slower, lower-amplitude
gait**, which the global parameterisation *can* represent.

Do not fit oracles with the leg damaged from rest: the real phase 2 is always
entered walking. The fit therefore damages at t = 1 s while walking, matching
the main experiment.

## Screen result (2026-07, RR leg, 60→22 Nm)

`results/damage_optima.json`, cross-penalty matrix (mean V over 6 seeds):

| gait | under healthy | under damaged |
|---|---|---|
| **incumbent** (flat-optimal) | +0.98 | **−2.00 (falls)** |
| damaged_opt (re-fit) | +0.90 | **+0.375 (walks)** |

**Screen gap = V(damaged_opt│damaged) − V(incumbent│damaged) = +2.375 → GO.**
The old CPG params fall under the weak leg; a re-optimised **in-box** gait
(weaker coupling + slower swing: `[4.0, 11.0, 25.0, 55.0, 0.39, 0.14, 0.5, 5.2]`)
walks. So there is genuine room for online adaptation. Confirmed in the full 60 s
protocol (4 seeds, damage at 30 s): no-adapt falls **4/4** at vx₂≈0.20; oracle
falls **2/4** and survives the rest at vx₂≈0.33 with lower tip deviation.

**Ceiling is bound-limited.** The strongest recovery reduces demanded torque by
dropping amplitude, and the incumbent already sits on the box floor
(`hip_amp=0.10`, `knee_amp=0.5` are the `cpg_bounds` lower bounds). Relaxing those
floors (hip_amp→0.03, knee_amp→0.20) lifts the best damaged gait to V≈+0.62
(vx≈0.29, 0 falls). So the standard box *can* clear the survive/fall boundary,
but the amplitude floor caps how well — the parameterisation partially limits the
gap, as the candidates note predicted for #3.

**Fit note.** Under the weak leg most of the box FALLS (flat V=−2 plateau), so a
naive GP-UCB never escapes the incumbent probe (the first fit reported a false
NO-GO). `fit_damage_oracles.py` therefore (1) shapes the fall score by survival
time to give BO a gradient out of the plateau (guidance only; true V reported),
and (2) fits `healthy` first and seeds the `damaged` fit with the healthy
optimum, a known in-box survivor. Re-run the fit whenever `--leg` /
`--damage-force` change.
