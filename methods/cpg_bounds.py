"""Shared 8D CPG parameter bounds.

Order (joint-space Righetti-style CPG, used by the main grid/BO/MARX-EFE
comparison): [coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN,
              hip_amplitude, knee_amplitude, b].

NOTE: The Cartesian foot-trajectory + IK constants below (ALPHA_HOPF, PHI_TROT,
leg_ik, …) belong to the Zhang et al. IROS 2024 Cartesian controller, which was
found to be laterally unstable on Laikago under position control (rolls over in
1-3 s even with hand-tuned params). The main comparison reverted to the
joint-space mapping above; those constants are retained only for the exploratory
`gpefe_optimizer`, which is not part of the flat/sloped/friction comparison.
"""

import numpy as np
import torch

# ── Parameter bounds (joint-space CPG) ────────────────────────────────────────
bounds = torch.tensor([
    [4.0, 10.0, 10.0, 25.0, 0.05, 0.10, 0.5, 0.1],   # lower
    [12.0, 25.0, 25.0, 60.0, 0.5, 0.35, 1.0, 10.0],  # upper
], dtype=torch.double)

bounds_lower = bounds[0]
bounds_upper = bounds[1]

# ── Hopf oscillator constants ─────────────────────────────────────────────────
ALPHA_HOPF = 50.0   # radial convergence rate (fast; keeps r ≈ μ within a few steps)

# Trot-gait phase-lag matrix Φ, leg order FL=0, FR=1, RL=2, RR=3.
# φ[i,j] is the desired phase lead of leg j over leg i (θ*_j − θ*_i).
# Trot pairs: (FL, RR) and (FR, RL) each π apart.
PHI_TROT = np.array([
    [ 0,        np.pi,   np.pi,   0      ],  # FL
    [-np.pi,    0,       0,      -np.pi  ],  # FR
    [-np.pi,    0,       0,      -np.pi  ],  # RL
    [ 0,        np.pi,   np.pi,   0      ],  # RR
], dtype=float)

# Initial phases: FL/RR in stance (sin < 0), FR/RL in swing (sin > 0).
THETA_TROT_INIT = np.array([3 * np.pi / 2,  # FL — stance
                             np.pi / 2,       # FR — swing
                             np.pi / 2,       # RL — swing
                             3 * np.pi / 2])  # RR — stance

# ── Laikago leg geometry ──────────────────────────────────────────────────────
H_LEG  = 0.33   # nominal vertical foot-to-hip distance at neutral pose [m]
D_STEP = 0.05   # foot-trajectory step half-length [m] (fixed, as in Zhang et al.)

# Empirical IK constants derived from PyBullet FK measurements on laikago_toes.urdf.
# The 3D geometry (lateral offsets, RPY chain) means the effective planar 2-link
# model has A = L1²+L2², B = 2·L1·L2, and a knee-phase offset δ = -0.457 rad.
_IK_A   = 0.10294   # L1²+L2²
_IK_B   = 0.0972    # 2·L1·L2
_IK_D   = 0.457     # knee-offset: q_knee_eff = q_knee + δ where δ = -D
_IK_C0  = -0.288    # intercept of foot-angle–vs–q_knee regression
_IK_C1  = -0.601    # slope (d(foot_angle_intercept)/d(q_knee))


def leg_ik(fx, fz):
    """Empirical 2-link planar IK for a Laikago leg (laikago_toes.urdf).

    Parameters
    ----------
    fx : float  forward displacement of foot from hip joint [m]  (+ = forward,
                = Z-axis in PyBullet world frame for this URDF)
    fz : float  vertical displacement of foot from hip joint [m] (negative = below)

    Returns
    -------
    q_hip  : float  hip-forward joint angle [rad]
    q_knee : float  knee joint angle [rad]  (always ≤ 0 = bent)
    """
    d2 = fx * fx + fz * fz
    cos_arg = np.clip((d2 - _IK_A) / _IK_B, -1.0, 1.0)
    q_knee  = _IK_D - np.arccos(cos_arg)
    q_hip   = (_IK_C0 + _IK_C1 * q_knee) - np.arctan2(fx, -fz)
    return q_hip, q_knee
