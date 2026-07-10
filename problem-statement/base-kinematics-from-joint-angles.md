# Base motion of a quadruped as a function of joint angles

**Short answer up front:** the base pose of a legged robot is *not* a pure
function of the joint angles alone. Unlike a fixed-base manipulator, a quadruped
is a *floating base*: it has 6 unactuated degrees of freedom. Joint angles `q`
only tell you where the feet are *relative to the base*. To recover how the base
moves in the world you additionally need to know **which feet are in contact**,
and you exploit the constraint that a foot in stance does not move (zero
velocity / fixed position). This is exactly what "leg odometry" does.

## Setup / notation

- `q` — joint angles (per leg `ℓ`: `q_ℓ`), from encoders.
- `x_ℓ^b(q_ℓ)` — position of foot `ℓ` in the **base frame**, from *forward
  kinematics* (a pure function of joint angles).
- `J_ℓ(q_ℓ) = ∂x_ℓ^b/∂q_ℓ` — the leg Jacobian.
- `ω^b` — base angular velocity (in practice from the IMU gyroscope).
- `α_ℓ ∈ {0,1}` — contact flag for leg `ℓ`; `n_s` = number of stance legs.
- `R^{wb}` — rotation from base to world frame.

## 1. Angular position of the base

Joint angles give **no** information about base orientation: the leg kinematics
are invariant to how the whole body is rotated in the world. So

```
Ṙ^{wb} = R^{wb} [ω^b]_×,     ω^b from the gyroscope, not from q.
```

Orientation must come from an IMU (or an absolute sensor). This is why every
legged-robot state estimator fuses proprioceptive kinematics with an IMU.

## 2. Linear velocity of the base (the part that *does* use joint angles)

Assume a stance foot has zero world velocity (no slip). Differentiating
`x^w_foot = x^w_base + R^{wb} x_ℓ^b(q_ℓ)` and setting the foot velocity to zero
gives each stance leg's estimate of the base velocity, expressed in the base
frame:

```
v_ℓ^b = -α_ℓ ( J_ℓ(q_ℓ) q̇_ℓ + ω^b × x_ℓ^b(q_ℓ) )
```

The term `J_ℓ q̇_ℓ` is the foot's velocity due to joint motion; `ω^b × x_ℓ^b`
removes the rotational contribution. Averaging over stance legs:

```
v^b = (1 / n_s) Σ_{ℓ ∈ stance} v_ℓ^b
```

Then integrate in the world frame for **linear position**:

```
ṗ^w_base = R^{wb} v^b,     p^w_base(t) = p^w_base(0) + ∫_0^t R^{wb} v^b dt.
```

## 3. Equivalent discrete / pose form

Instead of velocities you can propagate the *relative* forward kinematics
between the foot currently in contact and the next foot to make contact. If leg
`ℓ` stays planted between `t` and `t+1`, the base displacement is recovered from
the change in `x_ℓ^b(q_ℓ)` under the fixed-foot constraint (the discrete version
of the same idea). This is the "forward-kinematic factor + contact factor"
formulation used in factor-graph estimators (Hartley et al.).

## Practical notes

- Leg odometry drifts (slippage, contact mis-detection, compliant feet), and it
  gives you **nothing** about yaw/orientation.
- In real systems these equations are always fused with an IMU in an EKF,
  invariant-EKF, or factor graph.
- If your goal is just the base's incremental motion from encoders, equations
  (2)–(3) are what you want; if you need absolute orientation you must add the
  IMU term in (1).

## Sources

- Fink & Semini, *Proprioceptive Sensor Fusion for Quadruped Robot State
  Estimation* (IIT-DLS):
  https://iit-dlslab.github.io/papers/fink20iros.pdf
- *MUSE: A Real-Time Multi-Sensor State Estimator for Quadruped Robots*:
  https://arxiv.org/html/2503.12101v1
- Hartley et al., *Legged Robot State-Estimation Through Combined Forward
  Kinematic and Preintegrated Contact Factors* (Univ. Michigan):
  http://robots.engin.umich.edu/publications/rhartley-2018a.pdf
- GTSAM tutorial, *Legged Robot Factors Part I*:
  https://gtsam.org/2019/09/18/legged-robot-factors-part-I.html
- *Multi-Sensor State Estimation Fusion on Quadruped Robot Locomotion*:
  https://arxiv.org/pdf/2007.02679
