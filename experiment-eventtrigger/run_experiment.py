"""Event-triggered CPG adaptation: GridSearch vs BO vs MARX-EFE, online.

The robot walks on flat ground with the flat-optimal CPG parameters and hits a
10 deg slope (placed at its t = 10 s position, as in experiment-flat2sloped).
A MARX model monitors the 0.5-s-ahead rollout prediction error (the reliable
terrain-change signal from demo-speedbump); when its EMA exceeds a calibrated
threshold, a parameter-adaptation EVENT fires and one of the methods starts
optimizing the CPG parameters online, in the same run, one candidate per
1.5 s window (0.3 s ramp, revert-to-best safeguard):

  noadapt  — keep the flat-optimal parameters (lower anchor),
  oracle   — ramp straight to the sloped-optimal parameters (upper anchor),
  grid     — Latin-hypercube sequence (the repo's grid-search baseline),
  bo       — GP-UCB on the windowed stability objective,
  marxefe  — expected-free-energy selection under the online MARX posterior
             (model updated every control step from t = 0, forgetting 0.995).

Hypothesis: the method that finds better parameters faster stays more stable —
the longer the robot walks the slope with mismatched parameters (high roll),
the harder the correction.

Stability objective (replaces the saturating velocity-tracking J): per window,

  J_stab = min(mean(vx)/v*, 1) − (RMS lp-roll + RMS detrended lp-pitch)[deg] / 10,

(lp = 0.5 s moving average, so the gait's rhythmic rocking is free)

with a fall scoring −2. Progress toward the target is rewarded but tolerant
(slower uphill is only mildly penalized, faster is not rewarded); attitude
dominates. Pitch is detrended by its window median so the slope's natural
pitch is free. Falls use a terrain-relative height criterion.

All arms of a seed share a bit-identical pre-trigger prefix (same jitter,
same flat parameters), so the trigger fires at the same step in every arm.

Usage (from repo root):
    python experiment-eventtrigger/run_experiment.py calibrate  # pick K
    python experiment-eventtrigger/run_experiment.py run [--seeds 100]
    python experiment-eventtrigger/run_experiment.py aggregate
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_HERE, "results")
_SEL_DIR = os.path.join(_REPO, "experiment-flat2sloped", "results")
SELECTED_JSON = os.path.join(_SEL_DIR, "selected_params.json")
TRIGGER_JSON = os.path.join(RESULTS_DIR, "trigger_config.json")
MAIN_CSV = os.path.join(RESULTS_DIR, "eventtrigger_experiment.csv")

# Terrain registry (RA-L review M6/C2). Each entry sets the incline used by the
# harness and the reference-optima file (incumbent = flat, oracle = that
# terrain's optimum). The sloped family reuses the whole pipeline unchanged
# except for the signed slope angle; friction / repeated transitions are added
# separately. `slope_deg` may be negative (decline).
TERRAINS = {
    "sloped10":  dict(kind="sloped", slope_deg=10.0,  selected="selected_params.json"),
    "incline15": dict(kind="sloped", slope_deg=15.0,  selected="selected_params_incline15.json"),
    "incline20": dict(kind="sloped", slope_deg=20.0,  selected="selected_params_incline20.json"),
    "decline10": dict(kind="sloped", slope_deg=-10.0, selected="selected_params_decline10.json"),
    # Friction drop: flat ground, grip falls from BASE_MU to ICE_MU at the
    # robot's t=10s position (placed like the slope, via find_y10). SLOPE_DEG=0
    # so the fall check reduces to the absolute-height criterion.
    "friction":  dict(kind="friction", slope_deg=0.0, selected="selected_params_friction.json"),
    # Repeated transitions: a staircase of +10° ramps separated by flat
    # landings, the first ramp at the robot's t=10s position. Tests the
    # "continual" claim and trigger re-arming (C2). Reuses the flat incumbent
    # and the 10° oracle (selected_params.json).
    "repeated":  dict(kind="repeated", slope_deg=10.0, selected="selected_params.json"),
}


def set_terrain(name):
    """Point the harness globals at terrain `name` (from TERRAINS). The trigger
    threshold K is terrain-dependent (recalibrated per terrain), so TRIGGER_JSON
    is keyed by terrain; sloped10 keeps the original trigger_config.json."""
    global SLOPE_DEG, TERRAIN_KIND, SELECTED_JSON, TRIGGER_JSON
    if name not in TERRAINS:
        raise SystemExit(f"unknown terrain {name!r}; choices: {list(TERRAINS)}")
    SLOPE_DEG = float(TERRAINS[name]["slope_deg"])
    TERRAIN_KIND = TERRAINS[name]["kind"]
    SELECTED_JSON = os.path.join(_SEL_DIR, TERRAINS[name]["selected"])
    suffix = "" if name == "sloped10" else f"_{name}"
    TRIGGER_JSON = os.path.join(RESULTS_DIR, f"trigger_config{suffix}.json")


def set_output_tag(tag):
    """Namespace the output CSV + figures + trigger config by `tag` (e.g. terrain
    name), so different terrains / ablation sub-runs do not overwrite each other.
    The empty tag keeps the original flat->10° filenames (trigger_config.json)."""
    global MAIN_CSV, FIG_TAG
    FIG_TAG = f"_{tag}" if tag else ""
    MAIN_CSV = os.path.join(RESULTS_DIR, f"eventtrigger_experiment{FIG_TAG}.csv")


FIG_TAG = ""

# ── Episode / terrain ────────────────────────────────────────────────────────
DT = 0.01
PRE_T = 10.0              # flat walking before the slope [s]
ADAPT_T = 20.0            # post-trigger horizon [s]
SLOPE_DEG = 10.0
TERRAIN_KIND = "sloped"  # "sloped", "friction", or "repeated" (set_terrain)
BASE_MU = 0.7            # friction terrain: grip before the drop (normal)
ICE_MU = 0.15            # friction terrain: grip after the drop (ice)
# Repeated-transition terrain: a staircase of +SLOPE_DEG ramps of REPEAT_RAMP_LEN
# separated by REPEAT_GAP_LEN flat landings; REPEAT_N ramps total. Each ramp
# onset is a terrain event the trigger should re-detect.
REPEAT_N = 2             # two +10° ramps (two transitions in one bout);
                         # the oracle survives both but falls if a 3rd is added,
                         # so two keeps the upper anchor meaningful
REPEAT_RAMP_LEN = 1.5
REPEAT_GAP_LEN = 1.5
CUR_SEGMENTS = None      # (y_start, slope_deg) list of the active repeated
                         # terrain, set in terrain_cfg; used by fallen_on_terrain
N_COLS = 1600
FAR_SLOPE_Y = 60.0
JITTER_STD = 0.002        # same seeding scheme as experiment-flat2sloped
ROBOT_MASS = 10.0
DEFAULT_ORI = [0.0, 0.5, 0.5, 0.0]
LEG_NAMES = ["FL", "FR", "RL", "RR"]

# ── Stability objective ──────────────────────────────────────────────────────
TARGET_VX = 0.5           # progress reference v* [m/s]; reward saturates above
ATT_REF_DEG = 10.0        # attitude normalization: 10 deg RMS costs 1 unit
J_FALL = -2.0
J_GOOD = 0.0              # window counts as "good parameters found"
                          # (progress reward exceeds sustained-attitude cost)

# ── Online adaptation ────────────────────────────────────────────────────────
WINDOW = 150              # steps per candidate window (1.5 s)
RAMP = 30                 # steps to ramp a new candidate in (0.3 s)
BO_N_RANDOM = 3           # random BO probes after the incumbent window
EFE_LAMBDA = 1e-2         # as methods.marxefe_optimizer.run_episode_maxrefe
SAFE_MARGIN = 0.15        # revert-to-best when a candidate scores this much
                          # below the best-known window score (same safety
                          # layer for grid / bo / marxefe; anchors exempt)

# ── Trigger monitor (demo-speedbump configuration) ───────────────────────────
H_PRED = 50               # rollout horizon [steps] (0.5 s)
WARMUP_UPDATES = 150
EMA_ALPHA = 0.08
BASELINE_T = (2.4, 3.4)   # flat window defining the per-run error baseline [s]
ARM_TIME = 3.5            # earliest allowed trigger [s]

ARMS = ["noadapt", "oracle", "grid", "bo", "marxefe"]


def terrain_cfg(change_y):
    """Terrain whose 'change' happens at forward position `change_y` (placed at
    the robot's t=10s position by find_y10, so every terrain fires the event in
    the same gait phase). Slope: ramp up/down at change_y. Friction: grip drops
    from BASE_MU to ICE_MU at change_y (ground stays flat)."""
    global CUR_SEGMENTS
    if TERRAIN_KIND == "friction":
        return {"kind": "friction",
                "zones": [(float(change_y), ICE_MU, "ice")],
                "base_mu": BASE_MU, "n_cols": N_COLS}
    if TERRAIN_KIND == "repeated":
        # Staircase: +SLOPE_DEG ramp, flat landing, repeated REPEAT_N times,
        # first ramp at change_y. Flat tail after the last ramp.
        segs, y = [], float(change_y)
        for _ in range(REPEAT_N):
            segs.append((round(y, 3), SLOPE_DEG))
            y += REPEAT_RAMP_LEN
            segs.append((round(y, 3), 0.0))
            y += REPEAT_GAP_LEN
        CUR_SEGMENTS = segs
        return {"kind": "multislope", "segments": segs, "n_cols": N_COLS}
    return {"kind": "sloped", "slope_deg": SLOPE_DEG,
            "slope_start_y": float(change_y), "n_cols": N_COLS}


def load_selected():
    """Return (incumbent = flat optimum, oracle = this terrain's optimum).
    The default flat->10° file uses key "sloped"; the per-terrain files written
    by gen_terrain_optima.py use key "oracle"."""
    with open(SELECTED_JSON) as f:
        sel = json.load(f)
    oracle = sel["oracle"] if "oracle" in sel else sel["sloped"]
    return (np.asarray(sel["flat"]["params"], float),
            np.asarray(oracle["params"], float))


def _lowpass(x, w=50):
    """Moving average over w steps (0.5 s): removes the gait's rhythmic
    oscillation (~2-3 Hz), keeps sustained lean / drift."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x
    w = min(w, len(x))
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty_like(x)
    for i in range(len(x)):
        a, b = max(0, i - w + 1), i + 1
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def j_stab_window(vx, roll, pitch, fell):
    """Stability objective over one window (higher is better). Attitude is
    low-passed first so the CPG's natural stride-frequency rocking is free;
    only sustained roll lean and within-window pitch drift are penalized."""
    if fell or len(vx) < 20:
        return J_FALL
    r_v = min(max(float(np.mean(vx)), 0.0) / TARGET_VX, 1.0)
    lp_roll = _lowpass(roll)
    rms_roll = np.rad2deg(np.sqrt(np.mean(lp_roll ** 2)))
    lp_pitch = _lowpass(pitch)
    rms_pitch = np.rad2deg(np.sqrt(np.mean((lp_pitch - np.median(lp_pitch)) ** 2)))
    return r_v - (rms_roll + rms_pitch) / ATT_REF_DEG


def _ground_height(y, slope_start_y):
    """Ground elevation at forward position `y` for the active terrain, so the
    height fall-check stays valid after the robot climbs. Repeated terrain uses
    the piecewise staircase profile; sloped uses a single ramp; friction/flat
    are level."""
    if TERRAIN_KIND == "repeated" and CUR_SEGMENTS:
        from methods.terrain import _piecewise_height_fn
        return float(_piecewise_height_fn(CUR_SEGMENTS)(np.array([float(y)]))[0])
    return np.tan(np.deg2rad(SLOPE_DEG)) * max(0.0, y - slope_start_y)


def fallen_on_terrain(base_pos, base_ori, slope_start_y, p):
    """Fall check with terrain-relative height (the absolute z < 0.25 criterion
    is blind once the robot has climbed; the orientation criterion is kept)."""
    rot = p.getMatrixFromQuaternion(base_ori)
    upright = np.dot([0, 0, 1], rot[6:])
    ground_z = _ground_height(base_pos[1], slope_start_y)
    return upright < 0.3 or (base_pos[2] - ground_z) < 0.25


# ── Trigger monitor ──────────────────────────────────────────────────────────

class TriggerMonitor:
    """MARX rollout-error monitor (demo-speedbump). Feed every step; fires when
    the EMA of the H_PRED-step prediction error exceeds
    baseline_mean + k_sigma * baseline_std (baseline from BASELINE_T).

    Re-arming (for repeated transitions): the trigger fires on the RISING edge
    of the error above the threshold, then disarms until the error falls back
    below baseline_mean + REARM_FRAC * k_sigma * baseline_std (hysteresis), so
    it re-fires once per transition rather than continuously. ``fired`` /
    ``fire_step`` record the FIRST event (starts adaptation, unchanged for
    single-transition terrains); ``fire_steps`` records every event."""

    REARM_FRAC = 0.5

    def __init__(self, n_steps, k_sigma):
        from methods.marxefe_optimizer import build_marx_agent
        np.random.seed(0)                    # deterministic tiny prior mean
        self.agent = build_marx_agent(target_velocity=1.0, forgetting=0.995)
        self.k_sigma = float(k_sigma)
        self.pred = np.full((n_steps + H_PRED, 4), np.nan)
        self.ema_log = np.zeros(n_steps)
        self.ema = 0.0
        self.fired = False
        self.fire_step = None
        self.fire_steps = []
        self.armed = True

    def step(self, k, t, y_new, applied):
        self.agent.update(y_new, applied)
        e = 0.0
        if np.isfinite(self.pred[k]).all():
            e = float(np.linalg.norm(y_new - self.pred[k]))
        if self.agent.n_updates > WARMUP_UPDATES:
            m_pred, _ = self.agent.predictions(
                np.tile(applied[:, None], (1, H_PRED)), time_horizon=H_PRED)
            if k + H_PRED < self.pred.shape[0]:
                self.pred[k + H_PRED] = m_pred[:, -1]
        self.ema = e if k == 0 else (1 - EMA_ALPHA) * self.ema + EMA_ALPHA * e
        self.ema_log[k] = self.ema

        if t >= ARM_TIME:
            b0, b1 = int(BASELINE_T[0] / DT), int(BASELINE_T[1] / DT)
            mu = self.ema_log[b0:b1].mean()
            sd = self.ema_log[b0:b1].std()
            thr = mu + self.k_sigma * sd
            if self.armed and self.ema > thr:
                self.fire_steps.append(k)
                self.armed = False
                if not self.fired:
                    self.fired = True
                    self.fire_step = k
            elif not self.armed and self.ema < mu + self.REARM_FRAC * self.k_sigma * sd:
                self.armed = True
        return self.fired

    def zscore_trace(self):
        b0, b1 = int(BASELINE_T[0] / DT), int(BASELINE_T[1] / DT)
        mu = self.ema_log[b0:b1].mean()
        sd = max(self.ema_log[b0:b1].std(), 1e-12)
        return (self.ema_log - mu) / sd


# ── Online adaptation methods ────────────────────────────────────────────────

class NoAdapt:
    """Anchor: keep the incumbent parameters (no safeguard wrapper)."""
    name = "noadapt"
    safeguarded = False

    def __init__(self, incumbent, oracle_params, box, seed):
        self.inc = incumbent

    def on_step(self, y_new, applied):
        pass

    def observe(self, params, window_J):
        pass

    def candidate(self):
        return self.inc


class Oracle(NoAdapt):
    """Anchor: go straight to the sloped-optimal parameters."""
    name = "oracle"

    def __init__(self, incumbent, oracle_params, box, seed):
        self.target = oracle_params

    def candidate(self):
        return self.target


class GridSearchOnline(NoAdapt):
    """The repo's grid baseline (methods.grid_search): a Latin-hypercube,
    space-filling candidate sequence, tried one window at a time. First window
    re-measures the incumbent on the new terrain (anchor, symmetric with BO)."""
    name = "grid"
    safeguarded = True

    def __init__(self, incumbent, oracle_params, box, seed):
        from scipy.stats.qmc import LatinHypercube
        self.inc = incumbent
        lo, hi = box
        lhs = LatinHypercube(d=8, seed=int(seed)).random(n=64)
        self.seq = lo + lhs * (hi - lo)
        self.i = -1

    def candidate(self):
        if self.i < 0:
            self.i = 0
            return self.inc
        cand = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return cand


class BOOnline(NoAdapt):
    """GP-UCB on the windowed stability objective (methods.bo_optimizer.
    BOOptimizer). Window 0 scores the incumbent (GP anchor), then BO_N_RANDOM
    random probes, then UCB proposals with a fast-decaying beta. All scored
    windows (including safeguard recovery windows) enter the GP."""
    name = "bo"
    safeguarded = True

    def __init__(self, incumbent, oracle_params, box, seed):
        import torch
        from methods.bo_optimizer import BOOptimizer, BetaSchedule
        self.inc = incumbent
        lo, hi = box
        self.bo = BOOptimizer(
            bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
            beta_schedule=BetaSchedule(beta_init=2.0, beta_min=0.5,
                                       n_decay_start=8, gamma=0.8),
            n_init=1 + BO_N_RANDOM, seed=int(seed))
        self.rng = np.random.default_rng(20_000 + int(seed))
        self.lo, self.hi = lo, hi
        self.t = 0

    def observe(self, params, window_J):
        self.bo._append(np.asarray(params, float), float(window_J))

    def candidate(self):
        if self.t == 0:
            cand = self.inc
        elif self.t <= BO_N_RANDOM:
            cand = self.rng.uniform(self.lo, self.hi)
        else:
            try:
                model = self.bo.fit_model()
                beta = self.bo.beta_schedule(self.t)
                cand = self.bo.from_unit(self.bo.suggest(model, beta))
            except Exception as e:
                print(f"[bo] GP fit/suggest failed ({e}); random fallback")
                cand = self.rng.uniform(self.lo, self.hi)
        self.t += 1
        return np.asarray(cand, float)


class MarxEFE(NoAdapt):
    """Active-inference selection: the MARX posterior is updated every control
    step from t = 0 (forgetting 0.995 so the flat->slope change is tracked);
    at the trigger and every window boundary the 8 CPG parameters are
    re-selected by minimizing expected free energy. The control prior is
    centred on the incumbent parameters (prior preference for the current
    gait) rather than mid-bounds.

    Three class attributes parameterize the ablations (M5/C5 of the RA-L
    review); the subclasses below flip exactly one each:
      FORGETTING   — exponential forgetting factor (0.995 tracks the change;
                     1.0 = no forgetting).
      EPISTEMIC    — include the information-seeking EFE terms (True = full
                     active inference; False = greedy-MAP predictive selector).
      PRIOR_CENTER — control-prior mean: "incumbent" (prefer the current gait)
                     or "mid" (mid-bounds, no preference for staying put)."""
    name = "marxefe"
    safeguarded = True
    FORGETTING = 0.995
    EPISTEMIC = True
    PRIOR_CENTER = "incumbent"

    def __init__(self, incumbent, oracle_params, box, seed):
        from methods.marxefe_optimizer import build_marx_agent
        from methods.cpg_bounds import bounds_lower, bounds_upper
        np.random.seed(1)
        self.agent = build_marx_agent(
            target_velocity=TARGET_VX, control_prior_scale=0.15,
            goal_prior_std=(np.sqrt(0.5), np.sqrt(0.5),
                            np.deg2rad(45), np.deg2rad(45)),
            time_horizon=2, forgetting=self.FORGETTING)
        if self.PRIOR_CENTER == "mid":
            # build_marx_agent already centres μ at mid-bounds; keep it.
            self.agent.μ = 0.5 * (bounds_lower.numpy() + bounds_upper.numpy())
        else:
            self.agent.μ = np.asarray(incumbent, float).copy()
        lo, hi = box
        self.lims = [(float(lo[i]), float(hi[i]))
                     for _ in range(self.agent.thorizon) for i in range(8)]
        self.lo, self.hi = lo, hi
        self.inc = incumbent

    def on_step(self, y_new, applied):
        self.agent.update(y_new, applied)

    def candidate(self):
        try:
            u = self.agent.minimizeEFE(control_lims=self.lims,
                                       lambda_energy=EFE_LAMBDA,
                                       max_iter=100, tol=1e-3,
                                       epistemic=self.EPISTEMIC)
            return np.clip(np.asarray(u[:8], float), self.lo, self.hi)
        except Exception as e:
            print(f"[{self.name}] EFE failed ({e}); holding parameters")
            return None


class MarxEFEGreedy(MarxEFE):
    """Ablation (M5): drop the information-seeking EFE terms -> greedy-MAP
    predictive selector. Isolates what the active-inference machinery adds."""
    name = "marxefe_greedy"
    EPISTEMIC = False


class MarxEFENoForget(MarxEFE):
    """Ablation (M5): no forgetting (λ = 1). Tests whether tracking the
    non-stationary flat->slope change needs the forgetting factor."""
    name = "marxefe_noforget"
    FORGETTING = 1.0


class MarxEFEMidPrior(MarxEFE):
    """Ablation (M3/C5): control prior centred at mid-bounds instead of the
    incumbent. Tests whether MARX-EFE's caution comes from a prior preference
    for staying near the current gait."""
    name = "marxefe_midprior"
    PRIOR_CENTER = "mid"


class MarxEFENoSafeguard(MarxEFE):
    """Ablation (C5, the decisive adaptation test): MARX-EFE with the shared
    revert-to-best safeguard DISABLED. The safeguard restores the incumbent
    whenever a candidate underscores, which forces every arm back to the flat
    optimum by construction — so no method can demonstrate adaptation while it
    is active. With it off, MARX-EFE selects freely from its model every window;
    if its θ(t) then moves toward θ*(10°) while staying upright, it genuinely
    adapts (unlike grid/BO, which need destabilizing physical trials)."""
    name = "marxefe_nosafeguard"
    safeguarded = False


METHODS = {c.name: c for c in (NoAdapt, Oracle, GridSearchOnline,
                               BOOnline, MarxEFE, MarxEFEGreedy,
                               MarxEFENoForget, MarxEFEMidPrior,
                               MarxEFENoSafeguard)}


class Safeguard:
    """Uniform safety layer over the online optimizers: track the best-known
    (params, window J) pair — initialized with the incumbent and its score on
    the last pre-trigger window — and after any candidate window that scores
    below best_J − SAFE_MARGIN, spend the next window back at the best-known
    parameters (recovery). Recovery windows are scored and fed to the method
    like any other window; two recoveries never run back-to-back."""

    def __init__(self, method, incumbent, pre_trigger_J):
        self.m = method
        self.best_x = np.asarray(incumbent, float).copy()
        self.best_J = float(pre_trigger_J)
        self.last_x = None
        self.in_recovery = False

    def next_target(self, last_window_J):
        if last_window_J is not None and self.last_x is not None:
            self.m.observe(self.last_x, last_window_J)
            if last_window_J > self.best_J:
                self.best_J = float(last_window_J)
                self.best_x = self.last_x.copy()
        if (self.m.safeguarded and last_window_J is not None
                and not self.in_recovery
                and last_window_J < self.best_J - SAFE_MARGIN):
            self.in_recovery = True
            target = self.best_x
        else:
            self.in_recovery = False
            target = self.m.candidate()
        if target is not None:
            self.last_x = np.asarray(target, float).copy()
        return target


# ── Episode ──────────────────────────────────────────────────────────────────

def _reset_with_jitter(p, robot, seed):
    from methods.marxefe_optimizer import JointCPG
    rng = np.random.default_rng(10_000 + int(seed))
    jit = rng.normal(0.0, JITTER_STD, size=12)
    p.resetBasePositionAndOrientation(robot, [0.0, 0.0, 0.55], DEFAULT_ORI)
    p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])
    abd, hip, knee = [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14]
    for i, jid in enumerate(abd):
        p.resetJointState(robot, jid, 0.0 + jit[i])
    for i, jid in enumerate(hip):
        p.resetJointState(robot, jid, 0.05 + jit[4 + i])
    for i, jid in enumerate(knee):
        p.resetJointState(robot, jid, -0.6 + jit[8 + i])
    for _ in range(int(1.0 / DT)):
        for jid in abd:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
        for jid in hip:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, 0.25)
        for jid in knee:
            p.setJointMotorControl2(robot, jid, p.POSITION_CONTROL, -1.0)
        p.stepSimulation()
    return JointCPG(n_legs=4)


def run_trial(seed, slope_start_y, method_name, k_sigma,
              incumbent, oracle_params, box, duration):
    """One full episode: flat prefix -> trigger -> online adaptation.
    Returns logs including the per-window scores and the trigger step."""
    import pybullet as p
    from methods import terrain
    from methods.marxefe_optimizer import (extract_observation,
                                           get_base_orientation,
                                           load_environment, load_robot)

    terrain.TERRAIN_CONFIG = terrain_cfg(slope_start_y)
    load_environment(DT, use_gui=False)
    robot, _, joint_IDs_full, filtered, feet = load_robot(p)
    cpg = _reset_with_jitter(p, robot, seed)

    n_steps = int(round(duration / DT))
    monitor = TriggerMonitor(n_steps, k_sigma)
    method = METHODS[method_name](np.asarray(incumbent, float),
                                  np.asarray(oracle_params, float), box, seed)

    log = {k: np.zeros(n_steps) for k in ["y", "z", "vx", "roll", "pitch"]}
    seg_start = np.asarray(incumbent, float).copy()
    seg_target = seg_start.copy()
    seg_anchor = 0
    trigger_step = None
    guard = None                # built at the trigger (needs pre-trigger J)
    window_scores = []          # window J values after trigger
    selected_params = []        # the 8-vector applied in each post-trigger
                                # window (aligned with window_scores): what the
                                # method actually selected -> the θ(t) trajectory
    win_buf = {"vx": [], "roll": [], "pitch": []}
    fell, fall_step = False, None
    cur_y = 0.0                 # robot forward position (for friction terrain)

    for k in range(n_steps):
        t = k * DT

        # Window boundary handling (post-trigger): score the finished window,
        # ask the method for the next candidate, ramp toward it.
        if (trigger_step is not None and k > trigger_step
                and (k - trigger_step) % WINDOW == 0):
            last_J = j_stab_window(win_buf["vx"], win_buf["roll"],
                                   win_buf["pitch"], fell=False)
            window_scores.append(last_J)
            win_buf = {"vx": [], "roll": [], "pitch": []}
            target = guard.next_target(last_J)
            if target is not None:
                seg_start = applied.copy()
                seg_target = np.asarray(target, float)
                seg_anchor = k
            selected_params.append(np.asarray(seg_target, float).copy())

        frac = min(1.0, (k - seg_anchor) / max(1, RAMP))
        applied = seg_start + frac * (seg_target - seg_start)

        raw = np.array([int(len(p.getContactPoints(
            bodyA=0, bodyB=robot, linkIndexA=-1, linkIndexB=feet[j])) > 0)
            for j in range(4)])
        hips, knees = cpg.step(applied, raw, DT)
        for j in range(4):
            a_id, h_id, k_id = joint_IDs_full[LEG_NAMES[j]]
            p.setJointMotorControl2(robot, a_id, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j])
            p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j])
        # Friction terrain: set the ground grip for the robot's current position
        # (no-op on slope terrain). Uses last step's position (1-step lag).
        terrain.apply_dynamic_friction(p, robot, cur_y)
        p.stepSimulation()

        base_pos, base_ori = get_base_orientation(p, robot, DEFAULT_ORI)
        vel, _ = p.getBaseVelocity(robot)
        roll, pitch, _ = p.getEulerFromQuaternion(base_ori)
        cur_y = base_pos[1]
        log["y"][k], log["z"][k] = base_pos[1], base_pos[2]
        log["vx"][k] = vel[1]
        log["roll"][k], log["pitch"][k] = roll, pitch

        y_new = np.array([vel[1], vel[0], pitch, roll])
        fired = monitor.step(k, t, y_new, applied)
        method.on_step(y_new, applied)
        if fired and trigger_step is None:
            # Event: adapt immediately — first proposal ramps in from now.
            # The safeguard's best-known anchor is the incumbent scored on the
            # last pre-trigger window (its flat performance).
            trigger_step = monitor.fire_step
            k0 = max(0, k - WINDOW)
            pre_J = j_stab_window(log["vx"][k0:k + 1], log["roll"][k0:k + 1],
                                  log["pitch"][k0:k + 1], fell=False)
            guard = Safeguard(method, incumbent, pre_J)
            target = guard.next_target(None)
            if target is not None:
                seg_start = applied.copy()
                seg_target = np.asarray(target, float)
                seg_anchor = k + 1
            selected_params.append(np.asarray(seg_target, float).copy())
            win_buf = {"vx": [], "roll": [], "pitch": []}
        elif trigger_step is not None:
            win_buf["vx"].append(vel[1])
            win_buf["roll"].append(roll)
            win_buf["pitch"].append(pitch)

        if fallen_on_terrain(base_pos, base_ori, slope_start_y, p):
            fell, fall_step = True, k
            for key in log:
                log[key] = log[key][:k + 1]
            if trigger_step is not None:
                window_scores.append(J_FALL)
            break

        # Stop ADAPT_T after the trigger (episode budget reached).
        if trigger_step is not None and k - trigger_step >= int(ADAPT_T / DT):
            for key in log:
                log[key] = log[key][:k + 1]
            break

    p.disconnect()
    n = len(log["y"])
    return dict(log=log, fell=fell, fall_step=fall_step,
                trigger_step=trigger_step, window_scores=window_scores,
                fire_steps=[s for s in monitor.fire_steps if s < n],
                selected_params=[np.asarray(x, float).tolist()
                                 for x in selected_params],
                ema=monitor.ema_log[:n],
                z=monitor.zscore_trace()[:n])


def trial_metrics(res, slope_start_y):
    """Post-trigger metrics for one arm."""
    kT = res["trigger_step"]
    if kT is None:
        return dict(triggered=0)
    log = res["log"]
    n = len(log["y"])
    k1 = min(n, kT + int(ADAPT_T / DT))
    roll = log["roll"][kT:k1]
    pitch = log["pitch"][kT:k1]
    vx = log["vx"][kT:k1]
    fell = bool(res["fell"])
    surv = (res["fall_step"] - kT) * DT if fell else ADAPT_T

    # fraction of post-trigger time with sustained (low-passed) roll < 5 deg
    if len(roll) >= 20:
        frac_stable = float(np.mean(np.abs(np.rad2deg(_lowpass(roll))) < 5.0))
    else:
        frac_stable = 0.0

    ws = res["window_scores"]
    good = [i for i, J in enumerate(ws) if J >= J_GOOD]
    return dict(
        triggered=1,
        trigger_t=kT * DT,
        fell=int(fell),
        t_surv=float(surv),
        frac_stable=frac_stable,
        mean_J=float(np.mean(ws)) if ws else np.nan,
        best_J=float(np.max(ws)) if ws else np.nan,
        n_windows=len(ws),
        win_to_good=(good[0] if good else np.nan),
        dist=float(log["y"][min(k1, n) - 1] - log["y"][kT]),
        max_roll=float(np.rad2deg(np.max(np.abs(roll)))) if len(roll) else np.nan,
        mean_vx=float(np.mean(vx)) if len(vx) else np.nan,
        n_triggers=len(res.get("fire_steps", [])),
    )


# ── Per-seed calibration of the slope position (identical to flat2sloped) ────

def find_y10(seed, incumbent):
    res = run_trial(seed, FAR_SLOPE_Y, "noadapt", k_sigma=np.inf,
                    incumbent=incumbent, oracle_params=incumbent,
                    box=(incumbent, incumbent), duration=PRE_T)
    if res["fell"]:
        return None
    return float(res["log"]["y"][-1])


# ── Stage 1: trigger calibration ─────────────────────────────────────────────

def _calib_job(job):
    seed, incumbent, slope_deg, terrain_kind = job
    global SLOPE_DEG, TERRAIN_KIND
    SLOPE_DEG = float(slope_deg)
    TERRAIN_KIND = terrain_kind
    inc = np.asarray(incumbent, float)
    box = (inc, inc)
    y10 = find_y10(seed, inc)
    if y10 is None:
        return None
    # flat-only run (no slope in reach): z-trace for false positives
    flat = run_trial(seed, FAR_SLOPE_Y, "noadapt", k_sigma=np.inf,
                     incumbent=inc, oracle_params=inc, box=box,
                     duration=PRE_T + ADAPT_T)
    # flat->slope run without adaptation: z-trace for detection delay
    slope = run_trial(seed, y10, "noadapt", k_sigma=np.inf,
                      incumbent=inc, oracle_params=inc, box=box,
                      duration=PRE_T + ADAPT_T)
    return dict(seed=seed, y10=y10,
                z_flat=flat["z"], fell_flat=flat["fell"],
                z_slope=slope["z"], fell_slope=slope["fell"])


def calibrate(n_seeds, workers):
    incumbent, _ = load_selected()
    ctx = get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=2) as pool:
        out = pool.map(_calib_job, [(s, incumbent, SLOPE_DEG, TERRAIN_KIND)
                                    for s in range(n_seeds)])
    out = [o for o in out if o is not None]
    arm_k = int(ARM_TIME / DT)
    kT = int(PRE_T / DT)

    print(f"\ntrigger calibration on {len(out)} seeds "
          f"(EMA z-score of 0.5 s rollout error)")
    print(f"{'K':>5s} {'FP flat':>8s} {'miss':>6s} {'delay mean':>11s} "
          f"{'delay max':>10s}")
    chosen = None
    for K in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
        fp = sum(np.any(o["z_flat"][arm_k:] > K) for o in out)
        delays, miss = [], 0
        for o in out:
            zs = o["z_slope"]
            idx = np.where(zs[arm_k:] > K)[0]
            if len(idx) == 0:
                miss += 1
            else:
                delays.append((idx[0] + arm_k - kT) * DT)
        dm = np.mean(delays) if delays else np.nan
        dx = np.max(delays) if delays else np.nan
        print(f"{K:5.1f} {fp:>5d}/{len(out)} {miss:>4d} {dm:11.2f} {dx:10.2f}")
        if chosen is None and fp == 0 and miss == 0:
            chosen = dict(k_sigma=float(K), fp=int(fp), miss=int(miss),
                          delay_mean=float(dm), delay_max=float(dx))
    if chosen is None:
        raise RuntimeError("no K with 0 false positives and 0 misses; "
                           "inspect the z traces")
    # early-trigger check: any flat->slope trigger before the slope?
    early = sum(np.any(o["z_slope"][arm_k:kT] > chosen["k_sigma"]) for o in out)
    chosen["early_on_slope_runs"] = int(early)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TRIGGER_JSON, "w") as f:
        json.dump(chosen, f, indent=2)
    print(f"\nchosen K = {chosen['k_sigma']} "
          f"(mean delay {chosen['delay_mean']:.2f} s, "
          f"max {chosen['delay_max']:.2f} s, early triggers: {early})")
    print(f"saved {TRIGGER_JSON}")


# ── Stage 2: main experiment ─────────────────────────────────────────────────

def _seed_job(job):
    (seed, incumbent, oracle_params, k_sigma, trust_radius, arms,
     slope_deg, terrain_kind) = job
    # Spawned workers re-import the module with defaults; set the terrain this
    # run uses (terrain_cfg / fallen_on_terrain read these module globals).
    global SLOPE_DEG, TERRAIN_KIND
    SLOPE_DEG = float(slope_deg)
    TERRAIN_KIND = terrain_kind
    inc = np.asarray(incumbent, float)
    orc = np.asarray(oracle_params, float)

    from methods.cpg_bounds import bounds_lower, bounds_upper
    lb, ub = bounds_lower.numpy(), bounds_upper.numpy()
    if trust_radius > 0:
        lo = np.clip(inc - trust_radius * (ub - lb), lb, ub)
        hi = np.clip(inc + trust_radius * (ub - lb), lb, ub)
    else:
        lo, hi = lb, ub
    box = (lo, hi)

    y10 = find_y10(seed, inc)
    if y10 is None:
        return dict(seed=seed, valid=0, reason="fell_calibration")

    row = dict(seed=seed, valid=1, reason="", y10=y10,
               trust_radius=float(trust_radius))
    trig_steps = []
    for arm in arms:
        res = run_trial(seed, y10, arm, k_sigma, inc, orc, box,
                        duration=PRE_T + ADAPT_T + 5.0)
        m = trial_metrics(res, y10)
        if not m["triggered"]:
            row.update({"valid": 0,
                        "reason": f"no_trigger_{arm}"
                        if not res["fell"] else f"fell_pre_trigger_{arm}"})
            return row
        trig_steps.append(res["trigger_step"])
        for k, v in m.items():
            row[f"{arm}_{k}"] = v
        row[f"{arm}_windows"] = json.dumps(
            [round(float(x), 3) for x in res["window_scores"]])
        row[f"{arm}_params"] = json.dumps(
            [[round(float(v), 4) for v in p] for p in res["selected_params"]])
    row["trigger_spread"] = int(max(trig_steps) - min(trig_steps))
    return row


def run_main(seeds, workers, trust_radius, arms):
    incumbent, oracle_params = load_selected()
    with open(TRIGGER_JSON) as f:
        k_sigma = json.load(f)["k_sigma"]
    print(f"incumbent (flat-opt): {np.round(incumbent, 3).tolist()}")
    print(f"oracle (sloped-opt) : {np.round(oracle_params, 3).tolist()}")
    print(f"trigger K = {k_sigma}, trust_radius = {trust_radius}, arms = {arms}")

    jobs = [(s, incumbent, oracle_params, k_sigma, trust_radius, arms,
             SLOPE_DEG, TERRAIN_KIND) for s in range(seeds)]
    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(workers, maxtasksperchild=2) as pool:
        for i, row in enumerate(pool.imap_unordered(_seed_job, jobs)):
            rows.append(row)
            if row["valid"]:
                msg = (f"[{i+1:3d}/{seeds}] seed {row['seed']:3d} "
                       f"trig={row[f'{arms[0]}_trigger_t']:5.2f}s ")
                for a in arms:
                    st = "FELL" if row[f"{a}_fell"] else "ok"
                    msg += (f" {a}:{st} surv={row[f'{a}_t_surv']:4.1f}s "
                            f"J={row[f'{a}_mean_J']:5.2f}")
                print(msg, flush=True)
            else:
                print(f"[{i+1:3d}/{seeds}] seed {row['seed']:3d} INVALID "
                      f"({row['reason']})", flush=True)

    rows.sort(key=lambda r: r["seed"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(MAIN_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {MAIN_CSV}")


# ── Stage 3: aggregation ─────────────────────────────────────────────────────

def aggregate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(MAIN_CSV)
    n_total = len(df)
    df = df[df["valid"] == 1].copy()
    n = len(df)
    arms = [c[:-len("_fell")] for c in df.columns if c.endswith("_fell")]
    print(f"{n}/{n_total} seeds valid; trigger spread across arms: "
          f"max {df['trigger_spread'].max()} steps")
    trig = df[f"{arms[0]}_trigger_t"]
    print(f"trigger time: {trig.mean():.2f} +/- {trig.std():.2f} s "
          f"(slope reached at 10.0 s)")

    print(f"\n{'arm':10s} {'falls':>9s} {'surv[s]':>9s} {'fracStab':>9s} "
          f"{'meanJ':>7s} {'bestJ':>7s} {'win2good':>9s} {'dist[m]':>8s}")
    for a in arms:
        fe = df[f"{a}_fell"]
        print(f"{a:10s} {int(fe.sum()):>4d}/{n} ({100*fe.mean():3.0f}%) "
              f"{df[f'{a}_t_surv'].mean():>8.1f} "
              f"{df[f'{a}_frac_stable'].mean():>9.2f} "
              f"{df[f'{a}_mean_J'].mean():>7.2f} "
              f"{df[f'{a}_best_J'].mean():>7.2f} "
              f"{df[f'{a}_win_to_good'].median():>9.1f} "
              f"{df[f'{a}_dist'].mean():>8.2f}")

    print("\nfalls within 10 s of trigger (early phase; McNemar vs noadapt):")
    def _early(a):
        return (df[f"{a}_fell"] == 1) & (df[f"{a}_t_surv"] < 10.0)
    na = _early("noadapt")
    for a in arms:
        fa = _early(a)
        oa, ob = int((na & ~fa).sum()), int((~na & fa).sum())
        p = stats.binomtest(oa, oa + ob, 0.5).pvalue if oa + ob else np.nan
        print(f"  {a:8s}: {int(fa.sum()):3d}/{n}"
              + ("" if a == "noadapt" else f"   vs noadapt p = {p:.3f}"))

    print("\npairwise tests (McNemar on falls; Wilcoxon on survival time):")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            fa, fb = df[f"{a}_fell"].astype(bool), df[f"{b}_fell"].astype(bool)
            oa, ob = int((fa & ~fb).sum()), int((~fa & fb).sum())
            pm = (stats.binomtest(oa, oa + ob, 0.5).pvalue
                  if oa + ob > 0 else np.nan)
            try:
                pw = stats.wilcoxon(df[f"{a}_t_surv"], df[f"{b}_t_surv"]).pvalue
            except ValueError:
                pw = np.nan
            print(f"  {a:8s} vs {b:8s}: falls {int(fa.sum()):3d} vs "
                  f"{int(fb.sum()):3d}  McNemar p={pm:9.2e}   "
                  f"surv Wilcoxon p={pw:9.2e}")

    # ── Figure ───────────────────────────────────────────────────────────
    colors = {"noadapt": "#8a8984", "oracle": "#0b0b0b", "grid": "#2a9d64",
              "bo": "#2a78d6", "marxefe": "#eb6834"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    ax = axes[0, 0]           # survival curves
    tt = np.arange(0, ADAPT_T + DT, DT)
    for a in arms:
        surv = df[f"{a}_t_surv"].values
        frac = [(surv >= x).mean() for x in tt]
        ax.plot(tt, frac, color=colors.get(a, "k"), lw=2, label=a)
    ax.set_xlabel("time since trigger [s]")
    ax.set_ylabel("fraction upright")
    ax.set_ylim(0, 1.02)
    ax.set_title("Survival after the adaptation event")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]           # fall-rate bars
    bars = [100 * df[f"{a}_fell"].mean() for a in arms]
    ax.bar(arms, bars, color=[colors.get(a, "k") for a in arms], width=0.6)
    for i, b in enumerate(bars):
        ax.text(i, b + 0.7, f"{b:.0f}%", ha="center", fontsize=10)
    ax.set_ylabel("fall rate within 20 s of trigger [%]")
    ax.set_title("Falls per method")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]           # median window-score trajectory
    n_show = int(ADAPT_T / (WINDOW * DT))
    for a in arms:
        traj = np.full((n, n_show), np.nan)
        for r, (_, row) in enumerate(df.iterrows()):
            ws = json.loads(row[f"{a}_windows"])
            traj[r, :min(len(ws), n_show)] = ws[:n_show]
            if len(ws) < n_show and row[f"{a}_fell"]:
                traj[r, len(ws):] = J_FALL       # carry the fall forward
        med = np.nanmedian(traj, axis=0)
        q1 = np.nanpercentile(traj, 25, axis=0)
        q3 = np.nanpercentile(traj, 75, axis=0)
        x = np.arange(n_show)
        ax.plot(x, med, color=colors.get(a, "k"), lw=2, label=a)
        ax.fill_between(x, q1, q3, color=colors.get(a, "k"), alpha=0.12)
    ax.axhline(J_GOOD, color="#8a8984", lw=1, ls="--")
    ax.text(n_show - 0.5, J_GOOD + 0.03, "good", ha="right", fontsize=8,
            color="#52514e")
    ax.set_xlabel("window since trigger (1.5 s each)")
    ax.set_ylabel("window stability objective $J_{stab}$")
    ax.set_title("Adaptation speed (median, IQR band; falls carried at −2)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]           # per-seed mean J distributions
    data = [df[f"{a}_mean_J"].dropna().values for a in arms]
    bp = ax.boxplot(data, tick_labels=arms, showfliers=False, widths=0.55,
                    patch_artist=True, medianprops=dict(color="#0b0b0b"))
    for patch, a in zip(bp["boxes"], arms):
        patch.set_facecolor(colors.get(a, "k"))
        patch.set_alpha(0.55)
    ax.set_ylabel("mean post-trigger $J_{stab}$")
    ax.set_title("Post-trigger stability objective per seed")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Event-triggered CPG adaptation on flat→10° slope "
                 f"({n} seeds, adaptation window {ADAPT_T:.0f} s)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULTS_DIR, f"eventtrigger_comparison{FIG_TAG}.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")

    # θ(t) trajectory analysis (M3): does each method actually move toward the
    # incline optimum, or does it stay near the incumbent?
    if any(f"{a}_params" in df.columns for a in arms):
        plot_theta_trajectory(df, arms, plt)


PARAM_NAMES = ["coupling γ", "ω_swing", "ω_stance", "F_fast", "K_stop",
               "A_hip", "A_knee", "b"]


def plot_theta_trajectory(df, arms, plt):
    """θ(t) evidence for M3. For every method that selects parameters, parse the
    per-window selected θ, normalize to the CPG bounds, and quantify how far it
    moves from the flat incumbent θ*(flat) toward the incline optimum θ*(10°):

        gap_closed(window) = 1 − ‖θ_sel − θ*(10°)‖ / ‖θ*(flat) − θ*(10°)‖

    (0 = stayed at the incumbent, 1 = reached the incline optimum), all in the
    bound-normalized space. Produces (a) a gap-closed-vs-window curve per method
    and (b) an 8-panel per-parameter median trajectory with the two reference
    optima. Prints the final-window median gap closed per method."""
    import warnings
    import numpy as np
    from methods.cpg_bounds import bounds_lower, bounds_upper

    lb, ub = bounds_lower.numpy(), bounds_upper.numpy()
    span = np.where((ub - lb) > 0, ub - lb, 1.0)
    incumbent, oracle = load_selected()
    inc_n = (incumbent - lb) / span
    orc_n = (oracle - lb) / span
    gap = float(np.linalg.norm(inc_n - orc_n))
    if gap < 1e-9:
        print("\nθ(t): incumbent and oracle coincide; skipping gap-closed metric")
        return

    # The methods search a trust region of ±trust_radius of each parameter's
    # range around the incumbent, so θ*(10°) is generally NOT reachable. Report
    # the gap closed toward BOTH the true optimum and the best target achievable
    # inside the trust region (the true optimum clipped to the box) — otherwise a
    # perfect adapter looks like it "failed" only because the box caps its reach.
    tr = float(df["trust_radius"].iloc[0]) if "trust_radius" in df.columns else 0.0
    if tr > 0:
        orc_clip_n = np.clip(orc_n, inc_n - tr, inc_n + tr)
    else:
        orc_clip_n = orc_n
    gap_clip = max(float(np.linalg.norm(inc_n - orc_clip_n)), 1e-9)

    colors = {"grid": "#2a9d64", "bo": "#2a78d6", "marxefe": "#eb6834",
              "marxefe_greedy": "#f0a35e", "marxefe_noforget": "#9b59b6",
              "marxefe_midprior": "#16a085", "oracle": "#0b0b0b",
              "noadapt": "#8a8984"}
    selectors = [a for a in arms
                 if a not in ("noadapt", "oracle") and f"{a}_params" in df.columns]
    n_show = int(ADAPT_T / (WINDOW * DT))

    def arm_traj(a):
        """(n_seeds, n_show, 8) bound-normalized selected θ; NaN where missing."""
        T = np.full((len(df), n_show, 8), np.nan)
        for r, (_, row) in enumerate(df.iterrows()):
            try:
                ps = json.loads(row[f"{a}_params"])
            except (KeyError, TypeError, ValueError):
                continue
            for i, p in enumerate(ps[:n_show]):
                T[r, i, :] = (np.asarray(p, float) - lb) / span
        return T

    print("\nθ(t) gap closed toward θ*(10°) [and toward the trust-region-clipped "
          "optimum] (1 = reached, 0 = stayed at incumbent):")
    fig1, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(n_show)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for a in selectors:
            T = arm_traj(a)
            d = np.linalg.norm(T - orc_n[None, None, :], axis=2)  # (seeds, windows)
            closed = 1.0 - d / gap
            d_clip = np.linalg.norm(T - orc_clip_n[None, None, :], axis=2)
            closed_clip = 1.0 - d_clip / gap_clip
            med = np.nanmedian(closed, axis=0)
            q1 = np.nanpercentile(closed, 25, axis=0)
            q3 = np.nanpercentile(closed, 75, axis=0)
            ax.plot(x, med, color=colors.get(a, "k"), lw=2, label=a)
            ax.fill_between(x, q1, q3, color=colors.get(a, "k"), alpha=0.12)
            fin = med[np.isfinite(med)]
            mc = np.nanmedian(closed_clip, axis=0)
            finc = mc[np.isfinite(mc)]
            print(f"  {a:18s}: gap closed = "
                  f"{(fin[-1] if len(fin) else float('nan')):5.2f}   "
                  f"[clipped: {(finc[-1] if len(finc) else float('nan')):5.2f}]")
    ax.axhline(0.0, color="#8a8984", lw=1, ls="--")
    ax.axhline(1.0, color="#0b0b0b", lw=1, ls=":")
    ax.text(n_show - 0.5, 1.01, "θ*(10°)", ha="right", fontsize=8)
    ax.text(n_show - 0.5, 0.02, "θ*(flat) incumbent", ha="right", fontsize=8)
    ax.set_xlabel("window since trigger (1.5 s each)")
    ax.set_ylabel("fraction of incumbent→optimum gap closed")
    ax.set_title("Do the methods adapt θ toward the incline optimum? "
                 "(median, IQR band)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig1.tight_layout()
    out1 = os.path.join(RESULTS_DIR, f"theta_gap_closed{FIG_TAG}.png")
    fig1.savefig(out1, dpi=150)
    print(f"saved {out1}")

    # Per-parameter median trajectories (bound-normalized).
    trajs = {a: arm_traj(a) for a in selectors}
    fig2, axes = plt.subplots(2, 4, figsize=(15, 7))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j, axp in enumerate(axes.flat):
            for a in selectors:
                med = np.nanmedian(trajs[a][:, :, j], axis=0)
                axp.plot(x, med, color=colors.get(a, "k"), lw=1.6, label=a)
            axp.axhline(inc_n[j], color="#8a8984", lw=1, ls="--")
            axp.axhline(orc_n[j], color="#0b0b0b", lw=1, ls=":")
            if tr > 0:
                axp.axhline(orc_clip_n[j], color="#c0392b", lw=1, ls="-.")
            axp.set_title(PARAM_NAMES[j], fontsize=9)
            axp.set_ylim(-0.05, 1.05)
            axp.grid(alpha=0.3)
            if j == 0:
                axp.legend(fontsize=7)
    fig2.suptitle("Selected CPG parameters after the trigger (bound-normalized; "
                  "dashed = θ*(flat), dotted = θ*(10°), dash-dot = trust-region "
                  "limit toward θ*(10°))", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(RESULTS_DIR, f"theta_per_parameter{FIG_TAG}.png")
    fig2.savefig(out2, dpi=150)
    print(f"saved {out2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["calibrate", "run", "aggregate"])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--cal-seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--trust-radius", type=float, default=0.0,
                    help="0 = full CPG bounds (repo default); >0 = box of "
                         "this fraction of the bounds around the incumbent")
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--terrain", type=str, default="sloped10",
                    choices=list(TERRAINS),
                    help="terrain / reference-optima set (M6 generalization)")
    ap.add_argument("--tag", type=str, default=None,
                    help="output namespace for CSV+figures (default: terrain "
                         "name when not sloped10, else none)")
    args = ap.parse_args()

    set_terrain(args.terrain)
    tag = args.tag if args.tag is not None else (
        "" if args.terrain == "sloped10" else args.terrain)
    set_output_tag(tag)

    if args.stage == "calibrate":
        calibrate(args.cal_seeds, args.workers)
    elif args.stage == "run":
        run_main(args.seeds, args.workers, args.trust_radius,
                 args.arms.split(","))
    else:
        aggregate()


if __name__ == "__main__":
    main()
