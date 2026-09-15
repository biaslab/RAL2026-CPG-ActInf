"""Flat -> variable-slope online CPG adaptation, triggered by prediction error.

This experiment combines `experiment-flat2sloped` (the flat->slope terrain change
and its flat/slope CPG optima) with `experiment-eventtrigger` (the MARX-EFE goal
cross-entropy trigger and the squash-to-stop online adaptation protocol) into a
single scenario, matching the `problem-statement/` notebook:

  * The robot walks on flat ground with the FLAT-optimal CPG parameters. Five
    metres in, the terrain rises at a slope of variable degree (`--slopes`).
  * A MARX-EFE agent monitors the prediction error every control step: the goal
    cross-entropy between its one-step posterior predictive and a stable-walking
    goal prior. Normalised to a flat-walking baseline it is ~1 on the flat prefix
    and rises on the incline.
  * When that statistic exceeds K for >= 2 consecutive steps, an EVENT fires and
    an online method (`noadapt` / `oracle` / `grid` / `bo` / `marxefe`) starts
    searching for new CPG parameters, window by window. A shared revert-to-best
    safeguard protects the destructive optimisers.
  * SQUASH: adaptation runs only WHILE the prediction error is above K. Once the
    statistic falls back below K the method PAUSES (holds its parameters); a
    re-fire of the re-armed monitor resumes it. So the method stops once it has
    squashed its own prediction error.

Unlike experiment-eventtrigger (which writes aggregated metrics), this script
writes the FULL per-step signals of every run to `results/runs/*.npz` (robot
state, cross-entropy, ratio, applied CPG parameters, ...) plus a scalar
`results/manifest.csv`. The companion `analyze.ipynb` reads those files and
produces the comparison figures.

Usage (from repo root):
    python experiment-flat2slope-adapt/run_experiment.py run \
        --slopes 15 --seeds 5 --arms noadapt oracle grid bo marxefe --workers 8
    # then open experiment-flat2slope-adapt/analyze.ipynb
"""

import argparse
import csv
import json
import os
import sys
import time as _time
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from methods.cpg_bounds import bounds_lower, bounds_upper  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
MANIFEST_CSV = os.path.join(RESULTS_DIR, "manifest.csv")
CONFIG_JSON = os.path.join(RESULTS_DIR, "config.json")

# Flat / sloped CPG optima (same files the problem-statement notebook uses).
FLAT_JSON = os.path.join(_REPO, "experiment-flat", "results", "selected_params.json")
SLOPE_JSON = os.path.join(_REPO, "experiment-sloped", "results", "selected_params.json")

# ── Episode / terrain ────────────────────────────────────────────────────────
DT = 0.01
SLOPE_START_Y = 5.0       # the robot meets the incline 5 m in (spatial, fixed)
N_COLS = 1600             # heightfield forward extent (+/- 40 m)
JITTER_STD = 0.002        # initial joint-angle jitter [rad] (across-seed variation)
DEFAULT_ORI = [0.0, 0.5, 0.5, 0.0]
LEG_NAMES = ["FL", "FR", "RL", "RR"]
DURATION = 40.0           # hard cap on trial length [s]
ADAPT_T = 20.0            # stop this long after the trigger [s]

# ── Stability objective (per adaptation window) ──────────────────────────────
TARGET_VX = 0.5           # progress reference v* [m/s]; reward saturates above
ATT_REF_DEG = 10.0        # attitude normalisation: 10 deg RMS costs 1 unit
J_FALL = -2.0
J_GOOD = 0.0              # window counts as "good parameters found"

# ── Online adaptation ────────────────────────────────────────────────────────
WINDOW = 150              # steps per candidate window (1.5 s)
RAMP = 30                 # steps to ramp a new candidate in (0.3 s)
BO_N_RANDOM = 3           # random BO probes after the incumbent window
EFE_LAMBDA = 1e-2         # control-energy weight in the EFE selection
SAFE_MARGIN = 0.15        # revert-to-best margin below the best-known window J

# ── Trigger monitor (MARX-EFE goal cross-entropy) ────────────────────────────
WARMUP_UPDATES = 150      # steps before the model is trusted enough to report
EMA_ALPHA = 0.08          # smoothing of the raw cross-entropy
BASELINE_T = (2.4, 3.4)   # flat window defining the per-run baseline [s]
ARM_TIME = 3.5            # earliest allowed trigger [s]
MON_VEL_STD = 1.0         # goal-prior std on vx, vy [m/s] (speed: loose)
MON_PITCH_DEG = 20.0      # goal-prior std on pitch [deg] (tolerates slope offset)
MON_ROLL_DEG = 6.0        # goal-prior std on roll [deg] (lateral stability: tight)
K_DEFAULT = 2.0           # trigger threshold on the baseline-normalised ratio

# ── Decision-theoretic (optimality-based) trigger ────────────────────────────
# With the control cost off and the info-gain flat (identified model), the EFE at
# the incumbent is the goal cross-entropy, so triggering tests whether the CPG
# params are still at its minimum. The first-order statistic is the Newton
# decrement lambda^2 = g' H^+ g (g,H = grad/Hessian of the goal term w.r.t. u via
# the model's steady-state control gain). The decision-theoretic threshold fires
# when half*lambda^2 exceeds the control-prior cost of a reference gait move, i.e.
# lambda^2 > tau, tau = Du*(4*MOVE/CONTROL_PRIOR_SCALE)^2. Its ratio-to-tau plays
# the exact role the CE ratio-to-baseline plays, so the squash logic is unchanged
# (with the effective threshold = 1.0).
CONTROL_PRIOR_SCALE = 0.15   # monitor control-prior width (price of adapting)
MARX_CONTROL_PRIOR_SCALE = 0.15  # MarxEFE selection: control-prior width (pull
                                 # toward the incumbent); sigma = scale*range/4.
                                 # Overridable per experiment (module attribute).
MARX_GOAL_VEL_STD = MON_VEL_STD  # MarxEFE selection ONLY: goal-prior std on
                                 # vx,vy. Kept separate from MON_VEL_STD so an
                                 # experiment can tighten the method's velocity
                                 # goal without altering the trigger statistic.
MARX_PITCH_DEG = MON_PITCH_DEG   # MarxEFE selection ONLY: goal-prior std on
MARX_ROLL_DEG = MON_ROLL_DEG     # pitch/roll. Separate from MON_* so tightening
                                 # the method's attitude goal (the "brace" prior)
                                 # leaves the trigger monitors untouched.
DT_BUDGET_MOVE = 0.02        # reference gait move (fraction of each param's range)

# CUSUM variant of the decision-theoretic trigger (see CusumDecisionMonitor): the
# recoverable value lambda^2/tau is a per-step rate of value forgone by not
# adapting; the CUSUM accumulates it net of a tolerated baseline rate DT_CUSUM_SLACK
# and fires when the running sum crosses DT_CUSUM_H, so a SUSTAINED (transition)
# suboptimality accumulates and fires early while transient spikes / good-terrain
# noise do not. Units are fractions of the control-cost budget tau.
DT_CUSUM_SLACK = 0.10        # tolerated per-step suboptimality (kappa), in tau units
DT_CUSUM_H = 5.0             # CUSUM decision bound (h), in tau units

ARMS_DEFAULT = ["noadapt", "oracle", "grid", "bo", "marxefe"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _limit_threads():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def load_optima():
    """(flat-optimal, slope-optimal) 8-vectors."""
    with open(FLAT_JSON) as f:
        flat = np.asarray(json.load(f)["params"], float)
    with open(SLOPE_JSON) as f:
        slope = np.asarray(json.load(f)["params"], float)
    return flat, slope


def _lowpass(x, w=50):
    """0.5-s moving average: removes stride-frequency rocking, keeps drift."""
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
    low-passed first so the CPG's stride-frequency rocking is free; only
    sustained roll lean and within-window pitch drift are penalised."""
    if fell or len(vx) < 20:
        return J_FALL
    r_v = min(max(float(np.mean(vx)), 0.0) / TARGET_VX, 1.0)
    rms_roll = np.rad2deg(np.sqrt(np.mean(_lowpass(roll) ** 2)))
    lp_pitch = _lowpass(pitch)
    rms_pitch = np.rad2deg(np.sqrt(np.mean((lp_pitch - np.median(lp_pitch)) ** 2)))
    return r_v - (rms_roll + rms_pitch) / ATT_REF_DEG


def _ground_height(y, slope_deg):
    """Ground elevation at forward position y: flat until SLOPE_START_Y, then a
    single ramp at slope_deg."""
    return np.tan(np.deg2rad(slope_deg)) * max(0.0, y - SLOPE_START_Y)


def fallen_on_terrain(base_pos, base_ori, slope_deg, p):
    """Terrain-relative fall check (absolute-height test is blind once the robot
    climbs): fallen if it tips past upright<0.3 or its clearance drops below 0.25 m."""
    rot = p.getMatrixFromQuaternion(base_ori)
    upright = np.dot([0, 0, 1], rot[6:])
    clear = base_pos[2] - _ground_height(base_pos[1], slope_deg)
    return (upright < 0.3 or clear < 0.25), clear


# ── Trigger monitor ──────────────────────────────────────────────────────────

class TriggerMonitor:
    """MARX-EFE goal-cross-entropy monitor (ported from experiment-eventtrigger).

    Feed every step. The statistic is the cross-entropy between the agent's
    one-step posterior predictive p(y_k | u_k, D_{k-1}) and a stable-walking goal
    prior; EMA-smoothed and normalised to the flat baseline it is ~1 on flat and
    rises on the incline. Fires on the RISING edge above K (mean ratio) for
    PERSIST_STEPS steps, then re-arms once the ratio falls back below
    REARM_FRAC * K, so it re-fires once per transition (needed for squash)."""

    REARM_FRAC = 0.5
    PERSIST_STEPS = 2

    def __init__(self, n_steps, k_sigma):
        from methods.marxefe_optimizer import build_marx_agent
        np.random.seed(0)                      # deterministic tiny prior mean
        self.agent = build_marx_agent(
            target_velocity=TARGET_VX,
            goal_prior_std=(MON_VEL_STD, MON_VEL_STD,
                            np.deg2rad(MON_PITCH_DEG), np.deg2rad(MON_ROLL_DEG)),
            forgetting=0.995)
        self.k_sigma = float(k_sigma)
        self.c_log = np.zeros(n_steps)         # raw cross-entropy
        self.ema_log = np.zeros(n_steps)       # EMA cross-entropy
        self.ema = 0.0
        self.fired = False
        self.fire_step = None
        self.fire_steps = []
        self.armed = True
        self.above = 0

    def step(self, k, t, y_new, applied):
        c = 0.0
        if self.agent.n_updates > WARMUP_UPDATES:
            ubuf = self.agent.backshift(self.agent.ubuffer, applied)
            x_k = np.concatenate([ubuf.flatten(), self.agent.ybuffer.flatten()])
            try:
                c = float(self.agent.crossentropy(x_k))
            except Exception:
                c = 0.0
        self.agent.update(y_new, applied)
        self.ema = c if k == 0 else (1 - EMA_ALPHA) * self.ema + EMA_ALPHA * c
        self.c_log[k] = c
        self.ema_log[k] = self.ema

        if t >= ARM_TIME:
            thr = self.k_sigma * self._baseline_mean()
            self.above = self.above + 1 if self.ema > thr else 0
            if self.armed and self.above >= self.PERSIST_STEPS:
                self.fire_steps.append(k)
                self.armed = False
                if not self.fired:
                    self.fired = True
                    self.fire_step = k
            elif not self.armed and self.ema < self.REARM_FRAC * thr:
                self.armed = True
                self.above = 0
        return self.fired

    def _baseline_mean(self):
        b0, b1 = int(BASELINE_T[0] / DT), int(BASELINE_T[1] / DT)
        return max(self.ema_log[b0:b1].mean(), 1e-9)

    def ratio_trace(self):
        return self.ema_log / self._baseline_mean()

    def current_ratio(self, k):
        b0, b1 = int(BASELINE_T[0] / DT), int(BASELINE_T[1] / DT)
        if k < b1:
            return 0.0
        return float(self.ema_log[k] / self._baseline_mean())


def _dc_gain(agent):
    """Steady-state (DC) control gain B = (I - sum A_l)^-1 (sum B_c) of the learned
    ARX model, from the posterior mean M (columns follow the x layout
    [u-buffer | y-buffer]). The sustained effect of holding a new CPG parameter --
    the control authority relevant to adaptation, unlike the tiny one-step gain."""
    MT = agent.M.T
    Du, Dy = agent.Du, agent.Dy
    Wu, Wy = agent.ubuffer.shape[1], agent.ybuffer.shape[1]
    y_off = Du * Wu
    G_u = np.column_stack([MT[:, d*Wu:d*Wu+Wu].sum(1) for d in range(Du)])
    A = np.column_stack([MT[:, y_off+e*Wy:y_off+e*Wy+Wy].sum(1) for e in range(Dy)])
    return np.linalg.solve(np.eye(Dy) - A, G_u)


class DecisionTheoreticMonitor:
    """Optimality-based trigger. Same public interface as TriggerMonitor, but the
    statistic is the Newton decrement lambda^2 (recoverable goal cross-entropy at
    the incumbent controls) rather than the CE value, and the "baseline" it is
    normalised by is a fixed control-cost budget tau (not an empirical flat
    window). So `ratio_trace`/`current_ratio` return lambda^2 / tau and the
    trigger/squash logic fires/pauses against an effective threshold of 1.0."""

    REARM_FRAC = 0.5
    PERSIST_STEPS = 2

    def __init__(self, n_steps, k_sigma, dt_move=None, control_prior_scale=None):
        from methods.marxefe_optimizer import build_marx_agent
        move = DT_BUDGET_MOVE if dt_move is None else float(dt_move)
        cps = CONTROL_PRIOR_SCALE if control_prior_scale is None else float(control_prior_scale)
        np.random.seed(0)
        self.agent = build_marx_agent(
            target_velocity=TARGET_VX, control_prior_scale=cps,
            goal_prior_std=(MON_VEL_STD, MON_VEL_STD,
                            np.deg2rad(MON_PITCH_DEG), np.deg2rad(MON_ROLL_DEG)),
            forgetting=0.995)
        self.Du = self.agent.Du
        self.tau = self.Du * (4.0 * move / cps) ** 2      # control-cost budget
        self.k_sigma = float(k_sigma)                     # threshold on lambda^2/tau (=1.0)
        self.c_log = np.zeros(n_steps)                    # raw lambda^2
        self.ema_log = np.zeros(n_steps)                  # EMA lambda^2
        self.ctrl_log = np.zeros(n_steps)                 # model's Newton-step cost
        self.ema = 0.0
        self.fired = False
        self.fire_step = None
        self.fire_steps = []
        self.armed = True
        self.above = 0
        self._Sinv = np.linalg.inv(np.asarray(self.agent.goal_prior.cov, float))
        self._mstar = np.asarray(self.agent.goal_prior.mean, float)
        self._Ups = np.asarray(self.agent.Υ, float)

    def _lambda2(self, applied):
        ubuf = self.agent.backshift(self.agent.ubuffer, applied)
        x = np.concatenate([ubuf.flatten(), self.agent.ybuffer.flatten()])
        _, mu, _ = self.agent.posterior_predictive(x)
        err = mu - self._mstar
        B = _dc_gain(self.agent)
        g = B.T @ self._Sinv @ err
        H = B.T @ self._Sinv @ B
        Hp = np.linalg.pinv(H, rcond=1e-8)
        lam2 = float(g @ (Hp @ g))
        du = -Hp @ g
        ctrl = float(du @ (self._Ups @ du))
        return lam2, ctrl

    def step(self, k, t, y_new, applied):
        lam2, ctrl = 0.0, 0.0
        if self.agent.n_updates > WARMUP_UPDATES:
            try:
                lam2, ctrl = self._lambda2(applied)
            except Exception:
                lam2, ctrl = 0.0, 0.0
        self.agent.update(y_new, applied)
        self.ema = lam2 if k == 0 else (1 - EMA_ALPHA) * self.ema + EMA_ALPHA * lam2
        self.c_log[k] = lam2
        self.ema_log[k] = self.ema
        self.ctrl_log[k] = ctrl

        if t >= ARM_TIME:
            ratio = self.ema / self.tau
            self.above = self.above + 1 if ratio > self.k_sigma else 0
            if self.armed and self.above >= self.PERSIST_STEPS:
                self.fire_steps.append(k)
                self.armed = False
                if not self.fired:
                    self.fired = True
                    self.fire_step = k
            elif not self.armed and ratio < self.REARM_FRAC * self.k_sigma:
                self.armed = True
                self.above = 0
        return self.fired

    def _baseline_mean(self):
        return self.tau                       # fixed control-cost budget

    def ratio_trace(self):
        return self.ema_log / self.tau

    def current_ratio(self, k):
        return float(self.ema_log[k] / self.tau)


class CusumDecisionMonitor(DecisionTheoreticMonitor):
    """Decision-theoretic suboptimality trigger with a CUSUM decision rule.

    Same Newton-decrement lambda^2 as DecisionTheoreticMonitor (the goal cross-
    entropy that re-tuning the incumbent could recover), but instead of firing
    when the instantaneous EMA(lambda^2) exceeds the one-shot control-cost budget
    tau, it accumulates the recoverable value over time. Reading lambda^2/tau as
    the per-step rate of value forgone by holding the incumbent, the running sum

        S_k = max(0, S_{k-1} + (lambda^2_k / tau - kappa)),   fire when S_k > h,

    is the cumulative benefit of adapting net of a tolerated baseline rate kappa
    (DT_CUSUM_SLACK); it fires once that exceeds the decision bound h (DT_CUSUM_H).
    This is Page's sequential change detector applied to the optimality statistic:
    a SUSTAINED transition suboptimality (e.g. stepping onto a lower-friction band,
    where lambda^2 rises modestly for seconds) accumulates and fires early, while a
    transient one-step spike (a stumble) or good-terrain noise does not. On a truly
    un-recoverable surface (near-frictionless ice collapses the control gain, so
    lambda^2 stays low) it correctly stays silent.

    The lambda^2 / EMA / ctrl logs are inherited unchanged; only the fire/arm rule
    differs. `_baseline_mean` is redefined to tau*kappa so the shared squash logic
    (pause once the windowed ratio falls back below its threshold) pauses when the
    instantaneous suboptimality returns to the tolerated slack kappa."""

    def __init__(self, n_steps, k_sigma, dt_move=None, control_prior_scale=None,
                 cusum_slack=None, cusum_h=None):
        super().__init__(n_steps, k_sigma, dt_move, control_prior_scale)
        self.kappa = DT_CUSUM_SLACK if cusum_slack is None else float(cusum_slack)
        self.h = DT_CUSUM_H if cusum_h is None else float(cusum_h)
        self.S = 0.0
        self.s_log = np.zeros(n_steps)        # CUSUM statistic trace

    def step(self, k, t, y_new, applied):
        lam2, ctrl = 0.0, 0.0
        if self.agent.n_updates > WARMUP_UPDATES:
            try:
                lam2, ctrl = self._lambda2(applied)
            except Exception:
                lam2, ctrl = 0.0, 0.0
        self.agent.update(y_new, applied)
        self.ema = lam2 if k == 0 else (1 - EMA_ALPHA) * self.ema + EMA_ALPHA * lam2
        self.c_log[k] = lam2
        self.ema_log[k] = self.ema
        self.ctrl_log[k] = ctrl

        r = self.ema / self.tau                        # recoverable value, tau units
        if t >= ARM_TIME:
            if self.armed:
                self.S = max(0.0, self.S + (r - self.kappa))   # accumulate net of slack
                if self.S > self.h:
                    self.fire_steps.append(k)
                    self.armed = False
                    self.S = 0.0                        # reset accumulator on fire
                    if not self.fired:
                        self.fired = True
                        self.fire_step = k
            else:
                # disarmed: hold the accumulator down until the recoverable value
                # falls back to the tolerated baseline (suboptimality squashed),
                # then re-arm so the next transition can re-fire.
                self.S = 0.0
                if r < self.kappa:
                    self.armed = True
        self.s_log[k] = self.S
        return self.fired

    def _baseline_mean(self):
        return self.tau * self.kappa          # squash threshold = tolerated slack


# ── Online adaptation methods ────────────────────────────────────────────────

class NoAdapt:
    """Anchor: keep the incumbent (flat-optimal) parameters."""
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
    """Anchor: jump straight to the slope-optimal parameters."""
    name = "oracle"

    def __init__(self, incumbent, oracle_params, box, seed):
        self.target = oracle_params

    def candidate(self):
        return self.target


class GridSearchOnline(NoAdapt):
    """Latin-hypercube, space-filling candidate sequence, one window at a time.
    First window re-measures the incumbent on the new terrain (anchor)."""
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
    """GP-UCB on the windowed stability objective. Window 0 scores the incumbent
    (GP anchor), then BO_N_RANDOM random probes, then UCB proposals."""
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
    """Active-inference selection: the MARX posterior is updated every step
    (forgetting 0.995 so the flat->slope change is tracked); at each window
    boundary the 8 CPG parameters are re-selected by minimising expected free
    energy. The control prior is centred on the incumbent (prefer the current
    gait). Goal prior is the world-frame stability goal (pitch loose to tolerate
    the incline tilt, roll tight for lateral stability) -- the variant that was
    found to actually adapt while staying upright."""
    name = "marxefe"
    safeguarded = True
    FORGETTING = 0.995
    EPISTEMIC = True

    def __init__(self, incumbent, oracle_params, box, seed):
        from methods.marxefe_optimizer import build_marx_agent
        np.random.seed(1)
        goal_std = (MARX_GOAL_VEL_STD, MARX_GOAL_VEL_STD,
                    np.deg2rad(MARX_PITCH_DEG), np.deg2rad(MARX_ROLL_DEG))
        self.agent = build_marx_agent(
            target_velocity=TARGET_VX,
            control_prior_scale=MARX_CONTROL_PRIOR_SCALE,
            goal_prior_std=goal_std, time_horizon=2, forgetting=self.FORGETTING)
        self.agent.μ = np.asarray(incumbent, float).copy()   # prior = current gait
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
            print(f"[marxefe] EFE failed ({e}); holding parameters")
            return None


METHODS = {c.name: c for c in (NoAdapt, Oracle, GridSearchOnline, BOOnline, MarxEFE)}


class Safeguard:
    """Uniform safety layer over the online optimisers: track the best-known
    (params, window J), initialised with the incumbent and its last pre-trigger
    score; after any candidate window scoring below best_J - SAFE_MARGIN, spend
    the next window back at the best-known parameters (recovery). Recovery
    windows are scored and fed back like any other; two never run back-to-back."""

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


# ── One monitored, squash-adaptive episode ───────────────────────────────────

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


def run_trial(seed, slope_deg, method_name, k_sigma, incumbent, oracle_params, box,
              trigger="ce"):
    """One episode: flat prefix -> trigger -> squash-gated online adaptation.
    Logs the full per-step signals and returns them for NPZ dumping. `trigger`
    selects the monitor: "ce" (empirical goal cross-entropy ratio) or "dt"
    (decision-theoretic Newton-decrement lambda^2 vs a control-cost budget)."""
    import pybullet as p
    from methods import terrain
    from methods.marxefe_optimizer import (get_base_orientation,
                                           load_environment, load_robot)

    terrain.TERRAIN_CONFIG = {"kind": "sloped", "slope_deg": float(slope_deg),
                              "slope_start_y": SLOPE_START_Y, "n_cols": N_COLS}
    load_environment(DT, use_gui=False)
    robot, _, joint_IDs_full, _, feet = load_robot(p)
    cpg = _reset_with_jitter(p, robot, seed)

    n_steps = int(round(DURATION / DT))
    Monitor = DecisionTheoreticMonitor if trigger == "dt" else TriggerMonitor
    monitor = Monitor(n_steps, k_sigma)
    method = METHODS[method_name](np.asarray(incumbent, float),
                                  np.asarray(oracle_params, float), box, seed)

    keys = ["t", "x", "y", "z", "vx", "vy", "roll", "pitch", "clear"]
    log = {k: np.zeros(n_steps) for k in keys}
    applied_log = np.zeros((8, n_steps))
    adapting_log = np.zeros(n_steps, dtype=int)

    seg_start = np.asarray(incumbent, float).copy()
    seg_target = seg_start.copy()
    seg_anchor = 0
    applied = seg_start.copy()

    trigger_step = None
    guard = None
    window_scores = []
    selected_params = []
    win_buf = {"vx": [], "roll": [], "pitch": []}
    fell, fall_step = False, None
    adapting = True
    n_fires_seen = 0
    n_pauses = 0
    adapt_windows = 0
    propose_times = []
    roll = pitch = 0.0            # previous-step trunk attitude for CPG feedback

    for k in range(n_steps):
        t = k * DT

        # Window boundary (post-trigger): score the finished window, ask the
        # method for the next candidate, ramp toward it. SQUASH: pause proposals
        # once the ratio is back below K; resume on a re-fire of the monitor.
        if (trigger_step is not None and k > trigger_step
                and (k - trigger_step) % WINDOW == 0):
            last_J = j_stab_window(win_buf["vx"], win_buf["roll"],
                                   win_buf["pitch"], fell=False)
            window_scores.append(last_J)
            win_buf = {"vx": [], "roll": [], "pitch": []}
            # SQUASH: pause once the error is *sustainably* back to baseline, i.e.
            # the just-finished window's MEAN ratio is below K (a single-step dip
            # right after the trigger must not pause adaptation before it starts).
            win_ratio = (np.mean(monitor.ema_log[max(0, k - WINDOW):k])
                         / monitor._baseline_mean())
            # Allow a pause only after the method has made at least one genuine
            # (non-anchor) proposal, so "start adapting" always happens before
            # "stop once squashed" (the first post-trigger window replays the
            # incumbent for grid/bo, so pausing on it would collapse to noadapt).
            if adapting and win_ratio < k_sigma and adapt_windows >= 2:
                adapting = False
                n_pauses += 1
            elif not adapting and len(monitor.fire_steps) > n_fires_seen:
                adapting = True
            n_fires_seen = len(monitor.fire_steps)
            if adapting:
                t0 = _time.perf_counter()
                target = guard.next_target(last_J)
                propose_times.append(_time.perf_counter() - t0)
                adapt_windows += 1
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
        hips, knees = cpg.step(applied, raw, DT, roll=roll, pitch=pitch)
        for j in range(4):
            a_id, h_id, k_id = joint_IDs_full[LEG_NAMES[j]]
            p.setJointMotorControl2(robot, a_id, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
            p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j])
            p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()

        base_pos, base_ori = get_base_orientation(p, robot, DEFAULT_ORI)
        vel, _ = p.getBaseVelocity(robot)
        pitch, roll, _ = p.getEulerFromQuaternion(base_ori)  # physical (+Y fwd)
        fallen, clear = fallen_on_terrain(base_pos, base_ori, slope_deg, p)

        log["t"][k] = t
        log["x"][k], log["y"][k], log["z"][k] = base_pos[0], base_pos[1], base_pos[2]
        log["vx"][k], log["vy"][k] = vel[1], vel[0]
        log["roll"][k], log["pitch"][k], log["clear"][k] = roll, pitch, clear
        applied_log[:, k] = applied
        adapting_log[k] = int(adapting)

        y_new = np.array([vel[1], vel[0], pitch, roll])
        fired = monitor.step(k, t, y_new, applied)
        method.on_step(y_new, applied)

        if fired and trigger_step is None:
            trigger_step = monitor.fire_step
            k0 = max(0, k - WINDOW)
            pre_J = j_stab_window(log["vx"][k0:k + 1], log["roll"][k0:k + 1],
                                  log["pitch"][k0:k + 1], fell=False)
            guard = Safeguard(method, incumbent, pre_J)
            t0 = _time.perf_counter()
            target = guard.next_target(None)
            propose_times.append(_time.perf_counter() - t0)
            adapt_windows += 1
            n_fires_seen = len(monitor.fire_steps)
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

        if fallen:
            fell, fall_step = True, k
            if trigger_step is not None:
                window_scores.append(J_FALL)
            n_steps = k + 1
            break
        if trigger_step is not None and k - trigger_step >= int(ADAPT_T / DT):
            n_steps = k + 1
            break

    p.disconnect()

    n = n_steps
    for kk in keys:
        log[kk] = log[kk][:n]
    applied_log = applied_log[:, :n]
    adapting_log = adapting_log[:n]

    # Time-to-squash: first post-trigger step where the ratio is back below K.
    squash_step = None
    if trigger_step is not None:
        r_tr = monitor.ratio_trace()[:n]
        below = np.nonzero(r_tr[trigger_step + 1:] < k_sigma)[0]
        if len(below):
            squash_step = int(trigger_step + 1 + below[0])

    return dict(
        # per-step signals
        t=log["t"], x=log["x"], y=log["y"], z=log["z"],
        vx=log["vx"], vy=log["vy"], roll=log["roll"], pitch=log["pitch"],
        clear=log["clear"], applied=applied_log, adapting=adapting_log,
        ce_raw=monitor.c_log[:n], ce_ema=monitor.ema_log[:n],
        ratio=monitor.ratio_trace()[:n],
        dt_ctrl=(monitor.ctrl_log[:n] if trigger == "dt" else np.zeros(n)),
        trigger_kind=trigger,
        # events / summaries
        window_scores=np.asarray(window_scores, float),
        selected_params=np.asarray(selected_params, float) if selected_params
        else np.zeros((0, 8)),
        fire_steps=np.asarray([s for s in monitor.fire_steps if s < n], int),
        trigger_step=(-1 if trigger_step is None else int(trigger_step)),
        squash_step=(-1 if squash_step is None else int(squash_step)),
        fall_step=(-1 if fall_step is None else int(fall_step)),
        fell=int(fell),
        adapt_windows=int(adapt_windows),
        n_pauses=int(n_pauses),
        baseline_mean=float(monitor._baseline_mean()),
        # run identity / config
        seed=int(seed), slope_deg=float(slope_deg), method=method_name,
        k_sigma=float(k_sigma), incumbent=np.asarray(incumbent, float),
        oracle=np.asarray(oracle_params, float),
        slope_start_y=float(SLOPE_START_Y), dt=float(DT),
        window=int(WINDOW), ramp=int(RAMP), adapt_t=float(ADAPT_T),
        propose_t_total=float(np.sum(propose_times)) if propose_times else 0.0,
    )


# ── Scalar metrics for the manifest ──────────────────────────────────────────

def scalar_metrics(res):
    kT = res["trigger_step"]
    n = len(res["y"])
    triggered = int(kT >= 0)
    fell = int(res["fell"])
    fall_step = res["fall_step"]
    d = dict(seed=res["seed"], slope_deg=res["slope_deg"], method=res["method"],
             triggered=triggered, fell=fell,
             trigger_t=(kT * DT if triggered else np.nan),
             fall_t=(fall_step * DT if fall_step >= 0 else np.nan),
             squash_t=((res["squash_step"] - kT) * DT
                       if (triggered and res["squash_step"] >= 0) else np.nan),
             baseline_ce=res["baseline_mean"])
    if triggered:
        k1 = min(n, kT + int(ADAPT_T / DT))
        roll = res["roll"][kT:k1]
        surv = (fall_step - kT) * DT if fell else ADAPT_T
        ws = res["window_scores"]
        good = [i for i, J in enumerate(ws) if J >= J_GOOD]
        d.update(
            t_surv=float(surv),
            n_windows=int(len(ws)),
            mean_J=(float(np.mean(ws)) if len(ws) else np.nan),
            best_J=(float(np.max(ws)) if len(ws) else np.nan),
            win_to_good=(int(good[0]) if good else -1),
            n_proposals=int(res["adapt_windows"]),
            n_pauses=int(res["n_pauses"]),
            n_triggers=int(len(res["fire_steps"])),
            max_roll=(float(np.rad2deg(np.max(np.abs(roll)))) if len(roll) else np.nan),
            dist_on_slope=float(res["y"][min(k1, n) - 1] - res["y"][kT]),
        )
    else:
        d.update(t_surv=np.nan, n_windows=0, mean_J=np.nan, best_J=np.nan,
                 win_to_good=-1, n_proposals=0, n_pauses=0, n_triggers=0,
                 max_roll=np.nan, dist_on_slope=np.nan)
    return d


def _cond_suffix(trigger):
    """Filename suffix distinguishing trigger and attitude-feedback condition, so
    the CE/DT x feedback-on/off runs write to separate files."""
    suffix = "" if trigger == "ce" else f"_{trigger}"
    if os.environ.get("CPG_ATTITUDE_FB", "1") == "0":
        suffix += "_noafb"
    return suffix


def run_path(slope_deg, seed, method, trigger="ce"):
    suffix = _cond_suffix(trigger)
    return os.path.join(RUNS_DIR, f"slope{slope_deg:g}_seed{seed}_{method}{suffix}.npz")


# ── Job / harness ────────────────────────────────────────────────────────────

def _job(args):
    _limit_threads()
    slope_deg, seed, method, k_sigma, incumbent, oracle, trigger, dt_move = args
    global DT_BUDGET_MOVE
    DT_BUDGET_MOVE = float(dt_move)          # picked up by DecisionTheoreticMonitor
    from methods.marxefe_optimizer import JointCPG   # attitude-feedback ablation
    JointCPG.ATTITUDE_FEEDBACK = os.environ.get("CPG_ATTITUDE_FB", "1") != "0"
    from methods.cpg_bounds import bounds_lower as bl, bounds_upper as bu
    box = (bl.numpy(), bu.numpy())
    res = run_trial(seed, slope_deg, method, k_sigma, incumbent, oracle, box,
                    trigger=trigger)
    path = run_path(slope_deg, seed, method, trigger)
    np.savez_compressed(path, **res)
    row = scalar_metrics(res)
    row["trigger"] = trigger
    row["npz"] = os.path.relpath(path, RESULTS_DIR)
    return row


def run(slopes, seeds, arms, k_sigma, workers, trigger="ce", dt_move=DT_BUDGET_MOVE):
    os.makedirs(RUNS_DIR, exist_ok=True)
    incumbent, oracle = load_optima()
    # DT normalises lambda^2 by the control-cost budget tau, so its threshold on
    # the returned ratio is 1.0 (K is only the CE-ratio threshold).
    k_eff = 1.0 if trigger == "dt" else float(k_sigma)
    tau = 8 * (4.0 * dt_move / CONTROL_PRIOR_SCALE) ** 2
    suffix = _cond_suffix(trigger)
    manifest = MANIFEST_CSV if suffix == "" else os.path.join(
        RESULTS_DIR, f"manifest{suffix}.csv")
    config = CONFIG_JSON if suffix == "" else os.path.join(
        RESULTS_DIR, f"config{suffix}.json")

    print(f"flat-optimal (incumbent): {np.round(incumbent, 3).tolist()}")
    print(f"slope-optimal (oracle)  : {np.round(oracle, 3).tolist()}")
    print(f"trigger={trigger}  slopes={slopes} seeds={seeds} arms={arms} "
          f"threshold={k_eff}" + (f"  tau={tau:.3f} (move={dt_move})" if trigger == "dt" else ""))

    with open(config, "w") as f:
        json.dump(dict(trigger=trigger, slopes=slopes, seeds=seeds, arms=arms,
                       k_sigma=k_eff, dt_move=dt_move, tau=tau,
                       control_prior_scale=CONTROL_PRIOR_SCALE,
                       slope_start_y=SLOPE_START_Y, dt=DT, window=WINDOW,
                       ramp=RAMP, adapt_t=ADAPT_T, duration=DURATION,
                       target_vx=TARGET_VX,
                       mon_std_deg=dict(pitch=MON_PITCH_DEG, roll=MON_ROLL_DEG),
                       incumbent=incumbent.tolist(), oracle=oracle.tolist()),
                  f, indent=2)

    jobs = [(float(sl), int(s), m, k_eff, incumbent, oracle, trigger, dt_move)
            for sl in slopes for s in range(seeds) for m in arms]
    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=2) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(row)
            tg = "trig" if row["triggered"] else "NO-TRIG"
            print(f"[{i+1:3d}/{len(jobs)}] slope{row['slope_deg']:g} "
                  f"seed{row['seed']:>2} {row['method']:<8} {tg:>7}  "
                  f"fell={row['fell']} t_surv={row.get('t_surv', float('nan')):.1f} "
                  f"squash_t={row.get('squash_t', float('nan'))} "
                  f"meanJ={row.get('mean_J', float('nan')):.2f}", flush=True)

    rows.sort(key=lambda r: (r["slope_deg"], r["seed"], r["method"]))
    cols = ["slope_deg", "seed", "method", "trigger", "triggered", "fell",
            "trigger_t", "fall_t", "squash_t", "t_surv", "n_windows", "mean_J",
            "best_J", "win_to_good", "n_proposals", "n_pauses", "n_triggers",
            "max_roll", "dist_on_slope", "baseline_ce", "npz"]
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nsaved {manifest}  ({len(rows)} runs) and per-run NPZ in {RUNS_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--slopes", type=float, nargs="+", default=[15.0],
                    help="incline angles in degrees (one run set per value)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--arms", nargs="+", default=ARMS_DEFAULT,
                    choices=list(METHODS.keys()))
    ap.add_argument("--K", type=float, default=K_DEFAULT,
                    help="CE trigger threshold on the baseline-normalised ratio")
    ap.add_argument("--trigger", choices=["ce", "dt"], default="ce",
                    help="ce: goal cross-entropy ratio; dt: decision-theoretic "
                         "Newton-decrement lambda^2 vs a control-cost budget")
    ap.add_argument("--dt-move", type=float, default=DT_BUDGET_MOVE,
                    help="dt only: reference gait move (fraction of param range) "
                         "whose control cost sets the threshold tau")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-attitude-fb", action="store_true",
                    help="disable the CPG's VMC body-attitude feedback (ablation)")
    args = ap.parse_args()
    os.environ["CPG_ATTITUDE_FB"] = "0" if args.no_attitude_fb else "1"
    run(args.slopes, args.seeds, args.arms, args.K, args.workers,
        trigger=args.trigger, dt_move=args.dt_move)


if __name__ == "__main__":
    main()
