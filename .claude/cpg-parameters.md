# CPG parameters: the Hopf oscillator network

Reference note for the coupled-Hopf CPG used in this project.
Sources: `methods/cpg_controller.py`, `methods/cpg_bounds.py`, `main.tex` §III
(`sec:cpg`), hardware variant in `experiment-real/petoi_Hopf.py`.

## The oscillator

Four Hopf oscillators, one per leg, each a 2-D state $(x_i, y_i)$ on a limit
cycle of squared radius $u$ (`methods/cpg_controller.py:82-84`; paper at
`main.tex:167`):

$$\dot x_i = \alpha(u - r_i^2)x_i - \omega_i y_i$$
$$\dot y_i = \beta(u - r_i^2)y_i + \omega_i x_i + \gamma\sum_j k_{ij}y_j + s_i$$

with $r_i = \sqrt{x_i^2 + y_i^2}$ the instantaneous amplitude.

$x_i$ drives the hip (fore-aft swing), $y_i$ the knee (lift, only when
$y_i > 0$), so one oscillator produces one leg's step cycle:

- hip:  $\varphi^{hip}_i = \varphi^{hip}_0 + A_{hip} x_i$
- knee: $\varphi^{knee}_i = \varphi^{knee}_0 - A_{knee}\max(0, y_i) + \Delta_i$

## Fixed constants

`JointCPG` class attributes — never tuned:

| | value | role |
|---|---|---|
| `ALPHA` $\alpha$ | 3.0 | radial convergence of $x$ back to the limit cycle |
| `BETA` $\beta$ | 12.0 | same for $y$ — larger, so amplitude recovers faster in the lift direction |
| `U` $u$ | 2.0 | squared limit-cycle amplitude, $r_i \to \sqrt{2}$ |
| `K` | trot matrix | inhibitory coupling topology; pairs (FL,RR) and (FR,RL) antiphase |
| `HIP_OFFSET` | 0.26 rad | standing-pose hip offset |
| `KNEE_OFFSET` | -1.0 rad | standing-pose knee offset |
| `SWING_ENTER/EXIT` | 0.15 / 0.02 | phase-classification hysteresis on $y_i$ |
| `STANCE_ENTER/EXIT` | -0.15 / -0.02 | idem, stance side |
| `DEBOUNCE_THRESHOLD` | 2 ticks | contact-signal debounce |

## Tuned gait vector $\theta$ (8-D)

What the optimizer searches. Order matters — it is the order in
`methods/cpg_bounds.py:18` and in Eq. `eq:cpg-parameters` of the paper.

| # | symbol | bounds | what it does |
|---|---|---|---|
| 0 | $\gamma$ `coupling_gain` | 4 – 12 | strength of inter-limb coordination; holds the trot phasing against perturbation |
| 1 | $\omega_{swing}$ | 10 – 25 | angular frequency while the leg is in the air |
| 2 | $\omega_{stance}$ | 10 – 25 | frequency while loaded; the ratio to $\omega_{swing}$ sets **duty factor** |
| 3 | $F_{fast}$ | 25 – 60 | push applied when contact *contradicts* phase — accelerates the leg through to resync |
| 4 | $K_{stop}$ | 0.05 – 0.5 | cancels the oscillator's own drift when contact *matches* phase — effectively holds the leg |
| 5 | $A_{hip}$ | 0 – 0.35 rad | hip swing amplitude → **stride length** |
| 6 | $A_{knee}$ | 0 – 1.0 rad | knee flexion amplitude → **foot clearance** |
| 7 | $b$ | 0.1 – 10 | sigmoid sharpness of the swing/stance frequency switch; large $b$ = abrupt transition |

$\omega_i$ is not constant: it blends the two frequencies through a sigmoid,

$$\omega_i = \omega_{stance}\,\varsigma(-b y_i) + \omega_{swing}\,\varsigma(b y_i)$$

so a single oscillator runs fast in swing and slow in stance — that asymmetry
*is* the duty factor.

## The two feedback paths (not part of $\theta$)

**$s_i$ — Righetti contact feedback** (`cpg_controller.py:141-150`). Phase is
read off $y_i$ with hysteresis, contact is debounced over 2 ticks, and the two
are compared:

- contact *matches* phase (airborne in swing, loaded in stance):
  $s_i = K_{stop}(\omega_i x_i - \gamma\sum_j k_{ij}y_j)$
- otherwise (early touchdown, lost contact): $s_i = F_{fast}\,\mathrm{sign}(y_i)$

This resynchronizes the rhythm to the actual footfalls.

**$\Delta_i$ — attitude VMC** (`JointCPG.attitude_dknee`). A PD on trunk
roll/pitch added to the knee targets, distributed by leg geometry (`_FRONT`,
`_LEFT`) and clipped at `DKNEE_CLIP` = 0.35 rad. Pitch acts on deviation from a
slow EMA baseline so a steady incline is tolerated rather than fought; roll acts
on the raw angle. Its four gains `[kp_roll, kd_roll, kp_pitch, kd_pitch]`
(defaults 0.8, 0.05, 0.5, 0.05) are the **online-adaptable channel** via
`set_gains()` — continuous, no gait-phase discontinuity — as opposed to
$\theta$, which is held fixed during locomotion. Inactive unless `roll`/`pitch`
are passed to `step()`, so the open-loop CPG is the default.

## Variants

- **`PerLegCPG`** — 11-D, splits $A_{hip}$ into four per-leg amplitudes for the
  leg-damage experiment. `expand8()` / `expand_box()` lift the 8-D gait and box
  into 11-D. Everything else (oscillators, coupling, both feedback paths) is
  identical.
- **`experiment-real/petoi_Hopf.py`** — the Bittle hardware version: same
  dynamics, but $\alpha = \beta = 2.5$, $b = 100$ (near-hard switch), frequencies
  scaled by `eff_freq` (5 walk / 6 trot), separate walk and trot coupling
  matrices, and joint mapping in degrees
  (hip_offset 40, hip_amp 12, knee_offset 30, knee_amp 6).
- **`experiment-real/bittle_interface.py`** subclasses `JointCPG` and overrides
  only the CPG-state → joint-angle conversion, so the oscillator dynamics and
  both feedback paths are literally the same code in sim and on the robot.

## Caveat

`cpg_bounds.py` also holds `ALPHA_HOPF`, `PHI_TROT`, `THETA_TROT_INIT`, `H_LEG`,
`D_STEP` and `leg_ik`. These belong to the abandoned Zhang et al. IROS 2024
Cartesian foot-trajectory + IK controller (laterally unstable on Laikago under
position control), *not* to the joint-space CPG above. They are retained only for
the exploratory `gpefe_optimizer`.
