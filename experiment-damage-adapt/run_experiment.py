"""Continual leg-damage adaptation -- all five arms in one experiment.

The single, consolidated leg-damage experiment (supersedes the old
run_experiment / run_continual / run_gpsafe trio). One long, non-episodic bout on
flat ground:

  * the robot walks normally with all legs healthy;
  * after the first few seconds one hind leg (RR) is DAMAGED -- its hip+knee
    actuator maxForce ramps 60 -> 20 Nm over ~1 s, a persistent asymmetric
    under-actuation that fells the symmetric incumbent (4/4);
  * a prediction-error CUSUM detects the damage and the chosen METHOD responds;
  * the damage PERSISTS -- it is never auto-healed. Only a FALL reverts it: the
    robot is stood back upright at its position, the leg is HEALED, and after a
    random 2-8 s gap the damage re-engages (the next fall cycle). A method that
    adapts and does not fall keeps walking under the damage for the rest of the
    bout. Search-based methods carry their memory across recurrences.

Headline metric = FALLS PER BOUT: no-adapt tips over every ~time-to-fall seconds
and racks up many falls; a good adapter falls rarely.

PER-LEG CONTROL (the key to a fair leg-damage story): the CPG's hip amplitude is
split into one value per leg (methods.marxefe_optimizer.PerLegCPG, an 11-D control
vector), so the agent CAN compensate the asymmetric fault -- drop the weak leg's
swing amplitude and lean on the others. The no-adapt incumbent is this same
controller with all four hip amplitudes EQUAL (the symmetric flat-optimal gait),
which cannot express that compensation and so fails. With a single GLOBAL hip
amplitude there is no regime where no-adapt fails AND a recovery is findable
(the asymmetry is irreducible); per-leg control opens a broad, findable recovery.

Five arms (event_responders.ALL_ARMS), all searching the 4 per-leg hip amplitudes
(FREE_DIMS_DAMAGE) so the comparison is head-to-head:

  noadapt -> hold the symmetric flat-optimal gait (lower anchor; falls under damage);
  grid    -> Latin-hypercube proposals (naive search);
  bo      -> GP-UCB on the per-event stability score;
  safegp  -> the safe GP recovery agent (methods.gp_safe_agent);
  oracle  -> jump to the pre-fit per-leg recovery gait (upper anchor).

Per event (each ends at a fall, or at the bout end for a surviving gait) we
record fall / stability (RMS body tilt) / distance; results are written for the
analysis notebook (analyze.ipynb) to load:
  results/continual_events.csv   one row per event, tagged with `method`/`seed`
  results/continual_summary.csv  per-method aggregates (falls/bout, tilt, distance)
  results/logs/<method>_seed<k>.npz   per-seed step traces (incl. cumulative falls)

Usage (from repo root):
    python experiment-damage-adapt/run_experiment.py --seeds 5 --duration 120
    python experiment-damage-adapt/run_experiment.py --arms noadapt safegp oracle
    # oracle arm reads the per-leg recovery gait from results/damage_optima.json
    # (refit over the 4 per-leg hip amplitudes at the damage force)
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from methods import continual_driver as cd
from methods import event_responders as er
from methods import responder_worker as rw

RESULTS_DIR = os.path.join(_HERE, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
OPTIMA_JSON = os.path.join(RESULTS_DIR, "damage_optima.json")
INCUMBENT_JSON = os.path.join(RESULTS_DIR, "incumbent.json")

# PER-LEG control (11-D): the global hip amplitude is split into one per leg so an
# asymmetric leg fault can be compensated (see methods.marxefe_optimizer.PerLegCPG).
PARAM_NAMES = ["coupling", "w_swing", "w_stance", "F_FAST", "STOP",
               "hipA_FL", "hipA_FR", "hipA_RL", "hipA_RR", "kneeA", "b"]

# The flat-optimal incumbent (archived experiment-flat BO fit), 8-D global; kept
# here so the folder is self-contained; overridable via results/incumbent.json.
# Expanded to the 11-D per-leg layout (all four hip amplitudes equal) by
# load_incumbent -- so the no-adapt anchor is the symmetric gait that fails.
INCUMBENT8 = np.array([7.607, 13.0498, 25.0, 52.4044, 0.5, 0.1, 0.5, 10.0])

# safeGP/grid/bo search the 4 PER-LEG hip amplitudes (indices 5-8): the agent can
# drop the weak leg's swing amplitude and lean on the others -- the compensation
# the global-symmetric incumbent cannot express. (Feasibility 2026-07: at 20 Nm
# the global incumbent falls 4/4 while a per-leg gait recovers + travels ~10 m.)
FREE_DIMS_DAMAGE = [5, 6, 7, 8]

# ── Leg-damage scenario defaults (screened 2026-07; see README) ──────────────
DT = 0.01
LEG_NAMES = ["FL", "FR", "RL", "RR"]
DEFAULT_ORI = [0.0, 0.5, 0.5, 0.0]
DAMAGE_LEG = "RR"          # which leg's hip+knee weaken (a hind leg by default)
HEALTHY_FORCE = 60.0       # hip+knee maxForce before damage [Nm] (~ uncapped)
DAMAGE_FORCE = 20.0        # hip+knee maxForce under full damage [Nm]: at 20 the
                           # global-symmetric incumbent falls fast (4/4) while a
                           # per-leg gait can recover (per-leg CPG regime)
ABD_FORCE = 500.0          # abduction hold force [Nm]; left INTACT
DAMAGE_RAMP_T = 1.0        # torque droop ramped over this long [s] (no impulse)
SETTLE_STEPS = 60          # upright-reset settle [steps]
# The damage persists until a fall; EVAL_HOLD is only the trailing window (s)
# over which a SURVIVING gait's stability score V is measured for the responder's
# memory -- NOT an auto-heal timer.
EVAL_HOLD = 12.0

UPRIGHT_FALL = 0.30        # world-up . body-up below which = tip-over
HEIGHT_FALL = 0.25         # base height below which = collapsed [m] (flat ground)


def load_incumbent():
    """The 11-D per-leg incumbent: the symmetric flat-optimal gait (all four hip
    amplitudes equal). Sourced from results/incumbent.json (8-D) if present."""
    from methods.marxefe_optimizer import PerLegCPG
    p8 = (np.asarray(json.load(open(INCUMBENT_JSON))["params"], float)
          if os.path.exists(INCUMBENT_JSON) else INCUMBENT8)
    return PerLegCPG.expand8(p8)


def load_oracle_target():
    """Post-damage optimum (oracle arm), 11-D per-leg; None if not yet fit."""
    if not os.path.exists(OPTIMA_JSON):
        return None
    return np.asarray(json.load(open(OPTIMA_JSON))["damaged"]["params"], float)


class DamagePhysics:
    """Flat ground; one hind leg's hip+knee maxForce ramps from healthy_force to
    damage_force at event intensity `frac` (0 healthy .. 1 fully damaged).
    Implements the continual_driver physics contract."""

    def __init__(self, leg=DAMAGE_LEG, healthy_force=HEALTHY_FORCE,
                 damage_force=DAMAGE_FORCE):
        self.leg = leg
        self.hf, self.df = float(healthy_force), float(damage_force)
        self._p = None

    # ── physics contract ─────────────────────────────────────────────────────
    def setup(self, seed):
        import pybullet as p
        from methods import terrain
        from methods.marxefe_optimizer import (load_environment, load_robot,
                                               PerLegCPG)
        self._p = p
        PerLegCPG.ATTITUDE_FEEDBACK = True
        terrain.TERRAIN_CONFIG = {"kind": "flat"}
        load_environment(DT, use_gui=False)
        self.robot, _, self.jids, _, self.feet = load_robot(p)
        self.dmg_j = LEG_NAMES.index(self.leg)
        return self._settle([0.0, 0.0], seed)

    def actuate(self, cpg, applied, roll, pitch, frac):
        p = self._p
        legf = self.hf + frac * (self.df - self.hf)    # damaged-leg hip+knee force
        raw = np.array([int(len(p.getContactPoints(
            bodyA=0, bodyB=self.robot, linkIndexA=-1, linkIndexB=self.feet[j])) > 0)
            for j in range(4)])
        hips, knees = cpg.step(applied, raw, DT, roll=roll, pitch=pitch)
        for j in range(4):
            a_id, h_id, k_id = self.jids[LEG_NAMES[j]]
            p.setJointMotorControl2(self.robot, a_id, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=ABD_FORCE)
            if j == self.dmg_j:                         # weakened hip+knee motors
                p.setJointMotorControl2(self.robot, h_id, p.POSITION_CONTROL,
                                        hips[j], force=legf)
                p.setJointMotorControl2(self.robot, k_id, p.POSITION_CONTROL,
                                        knees[j], force=legf)
            else:                                       # healthy legs: default force
                p.setJointMotorControl2(self.robot, h_id, p.POSITION_CONTROL, hips[j])
                p.setJointMotorControl2(self.robot, k_id, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()
        return self._readout()

    def reset(self, at_xy, seed):
        return self._settle(at_xy, seed)

    def disconnect(self):
        if self._p is not None:
            self._p.disconnect()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _readout(self):
        from methods.marxefe_optimizer import get_base_orientation
        p = self._p
        base_pos, base_ori = get_base_orientation(p, self.robot, DEFAULT_ORI)
        vel, _ = p.getBaseVelocity(self.robot)
        pitch, roll, _ = p.getEulerFromQuaternion(base_ori)
        rot = p.getMatrixFromQuaternion(base_ori)
        upright = float(np.dot([0, 0, 1], rot[6:]))
        fell = (upright < UPRIGHT_FALL or base_pos[2] < HEIGHT_FALL)
        # measured hip+knee joint positions (for the AIF agent's exogenous input)
        ja = np.array([p.getJointState(self.robot, self.jids[L][j])[0]
                       for L in LEG_NAMES for j in (1, 2)], float)
        return cd.StepState(base_pos=base_pos, vx=vel[1], roll=roll, pitch=pitch,
                            fell=fell, vy=vel[0], joint_angles=ja)

    def _settle(self, at_xy, seed):
        """Stand the robot upright at (x,y); fresh CPG. The leg heals implicitly
        (event intensity frac ramps back to 0 -> force back to healthy_force)."""
        p = self._p
        from methods.marxefe_optimizer import PerLegCPG as JointCPG
        rng = np.random.default_rng(10_000 + int(seed))
        jit = rng.normal(0.0, 0.002, size=12)
        p.resetBasePositionAndOrientation(self.robot, [at_xy[0], at_xy[1], 0.55],
                                          DEFAULT_ORI)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        abd, hip, kn = [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14]
        for i, j in enumerate(abd):
            p.resetJointState(self.robot, j, 0.0 + jit[i])
        for i, j in enumerate(hip):
            p.resetJointState(self.robot, j, 0.05 + jit[4 + i])
        for i, j in enumerate(kn):
            p.resetJointState(self.robot, j, -0.6 + jit[8 + i])
        for _ in range(SETTLE_STEPS):
            for j in abd:
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                        targetPosition=0.0, force=ABD_FORCE)
            for j in hip:
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, 0.25)
            for j in kn:
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, -1.0)
            p.stepSimulation()
        return JointCPG(n_legs=4)


def build_spec(arm, incumbent, box, free, oracle_target, seed, args):
    """Picklable ResponderSpec for the out-of-process responder worker. The
    incumbent is pre-seeded as a known post-damage FALL (bo/safegp) inside the
    worker."""
    safegp_kwargs = dict(n_init=args.n_init, safe_V=args.safe_V, beta=args.beta,
                         kappa=args.kappa, objective=args.objective,
                         r_fall=args.r_fall,
                         efe_y_star=args.efe_y_star, efe_tau2=args.efe_tau2,
                         efe_adaptive=args.efe_adaptive,
                         efe_tau2_min=args.efe_tau2_min,
                         efe_tau2_max=args.efe_tau2_max)
    return rw.ResponderSpec(
        name=arm, incumbent=np.asarray(incumbent, float), box=box,
        free_dims=list(free), oracle_target=oracle_target, seed=int(seed),
        safegp_kwargs=safegp_kwargs, seed_fall=(arm in ("bo", "safegp")),
        v_fall=cd.BoutConfig().v_fall)


def run_seed(seed, arm, physics_kw, cfg, incumbent, box, free, oracle_target, args):
    physics = DamagePhysics(**physics_kw)
    if arm == "aif":                       # unified AIF agent: in-process, own trigger
        from methods.aif_recovery import UnifiedAIFAgent
        from methods.continual_driver_aif import run_event_bout_aif
        agent = UnifiedAIFAgent(incumbent, box, free, seed, dt=cfg.dt,
                                target_vx=cfg.target_vx)
        return run_event_bout_aif(seed, agent, physics, incumbent, cfg)
    spec = build_spec(arm, incumbent, box, free, oracle_target, seed, args)
    return cd.run_event_bout(seed, spec, physics, incumbent, cfg)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=er.ALL_ARMS,
                    choices=er.ALL_ARMS + ["aif"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed index (for slicing seeds across parallel "
                         "workers); runs seed_start .. seed_start+seeds-1")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for CSVs + per-seed logs (default "
                         "results/); inputs (oracle/incumbent JSON) still read "
                         "from results/. Use a per-worker dir for parallel runs")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--eval-hold", type=float, default=EVAL_HOLD,
                    help="trailing window [s] for scoring a SURVIVING gait's V "
                         "(memory feedback); NOT an auto-heal timer -- damage "
                         "persists until a fall regardless")
    ap.add_argument("--leg", default=DAMAGE_LEG, choices=["FL", "FR", "RL", "RR"],
                    help="which leg's hip+knee weaken (a HIND leg gives a real gap)")
    ap.add_argument("--healthy-force", type=float, default=HEALTHY_FORCE,
                    help="hip+knee maxForce before damage [Nm]")
    ap.add_argument("--damage-force", type=float, default=DAMAGE_FORCE,
                    help="hip+knee maxForce under full damage [Nm]")
    ap.add_argument("--free-dims", type=int, nargs="+", default=None,
                    help=f"CPG dims to search (default {FREE_DIMS_DAMAGE})")
    # safegp / bo agent knobs
    ap.add_argument("--n-init", type=int, default=4)
    ap.add_argument("--safe-V", type=float, default=-0.8)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--objective", choices=["ucb", "efe"], default="efe",
                    help="safegp planning objective: GP-UCB or Expected Free Energy")
    ap.add_argument("--r-fall", type=float, default=0.22,
                    help="UCB fall-exclusion radius (normalized) around remembered "
                         "falls; only used by objective=ucb. EFE has no such knob "
                         "(safety emerges from the goal prior)")
    ap.add_argument("--efe-y-star", type=float, default=1.0)
    ap.add_argument("--efe-tau2", type=float, default=0.5)
    ap.add_argument("--efe-adaptive", action="store_true")
    ap.add_argument("--efe-tau2-min", type=float, default=0.1)
    ap.add_argument("--efe-tau2-max", type=float, default=3.0)
    # detector / timing knobs
    ap.add_argument("--no-detector", action="store_true",
                    help="idealised: react at damage onset instead of detecting")
    ap.add_argument("--detect-kappa", type=float, default=0.15,
                    help="CUSUM slack (tolerated per-step prediction error)")
    ap.add_argument("--detect-h", type=float, default=1.8,
                    help="CUSUM decision threshold (fire when accumulator > h)")
    ap.add_argument("--detect-tau", type=float, default=0.4,
                    help="smoothing time constant for the health signal [s]")
    ap.add_argument("--gap-min", type=float, default=2.0)
    ap.add_argument("--gap-max", type=float, default=8.0)
    ap.add_argument("--grace", type=float, default=1.5,
                    help="post-heal grace before re-arming the detector [s]")
    ap.add_argument("--sim-speed", type=float, default=1.0,
                    help="real-time pacing factor: the sim is paced to sim_speed x "
                         "100 Hz so an optimizer taking T wall-seconds costs "
                         "~T*sim_speed*100 sim steps of latency. 1.0 is physically "
                         "faithful (a 120 s bout takes ~120 s wall); >1 runs faster "
                         "but inflates the latency penalty; <=0 disables pacing")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel bouts (ProcessPoolExecutor workers). Default "
                         "max(1, cpu-2). Each bout is real-time paced so mostly "
                         "idle; torch is pinned to 1 thread/responder so concurrent "
                         "GP fits don't fight over cores")
    a = ap.parse_args()

    out_results = a.out_dir if a.out_dir is not None else RESULTS_DIR
    out_logs = os.path.join(out_results, "logs")
    os.makedirs(out_results, exist_ok=True)
    os.makedirs(out_logs, exist_ok=True)
    import glob
    for _arm in a.arms:            # drop stale per-seed logs (e.g. a prior larger
        for _f in glob.glob(os.path.join(out_logs, f"{_arm}_seed*.npz")):  # --seeds run)
            os.remove(_f)
    incumbent = load_incumbent()
    oracle_target = load_oracle_target()
    if "oracle" in a.arms and oracle_target is None:
        raise SystemExit("oracle arm needs results/damage_optima.json "
                         "(run: python experiment-damage-adapt/fit_damage_oracles.py)")
    from methods.cpg_bounds import bounds_lower, bounds_upper
    from methods.marxefe_optimizer import PerLegCPG
    box = PerLegCPG.expand_box(bounds_lower.numpy(), bounds_upper.numpy())
    free = a.free_dims if a.free_dims is not None else FREE_DIMS_DAMAGE
    physics_kw = dict(leg=a.leg, healthy_force=a.healthy_force,
                      damage_force=a.damage_force)
    cfg = cd.BoutConfig(dt=DT, duration=a.duration, gap_min=a.gap_min,
                        gap_max=a.gap_max, event_ramp_t=DAMAGE_RAMP_T,
                        eval_hold=a.eval_hold,
                        use_detector=not a.no_detector, detect_tau=a.detect_tau,
                        detect_kappa=a.detect_kappa, detect_h=a.detect_h,
                        grace=a.grace, sim_speed=a.sim_speed)

    print(f"continual leg-damage adaptation: arms={a.arms} x {a.seeds} seeds "
          f"x {a.duration:g}s; {a.leg} {a.healthy_force:g}->{a.damage_force:g} Nm, "
          f"persists until a fall; re-engages {a.gap_min:g}-{a.gap_max:g}s after each")
    print(f"  incumbent: {np.round(incumbent, 3).tolist()}")
    print(f"  searching dims {free} ({[PARAM_NAMES[i] for i in free]})")
    det = ("idealised (react at onset)" if a.no_detector else
           f"prediction-error CUSUM (kappa={a.detect_kappa}, h={a.detect_h})")
    print(f"  detector: {det}\n")

    # ── run all (arm, seed) bouts in parallel (each is real-time paced) ───────
    njobs = a.jobs if (a.jobs and a.jobs > 0) else max(1, (os.cpu_count() or 4) - 2)
    seed_ids = list(range(a.seed_start, a.seed_start + a.seeds))
    tasks = [(arm, s) for arm in a.arms for s in seed_ids]
    est_wall = a.duration / a.sim_speed if a.sim_speed and a.sim_speed > 0 else 0.0
    print(f"  {len(tasks)} bouts on {njobs} parallel workers "
          f"(~{est_wall:.0f}s wall/bout at sim_speed={a.sim_speed:g}; "
          f"~{len(tasks) / njobs * est_wall / 60:.0f} min total)\n", flush=True)

    per = {}          # (arm, s) -> dict(events, dist, edist, nf, n_reset)
    failed = []
    done = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=njobs, mp_context=ctx) as ex:
        futs = {ex.submit(run_seed, s, arm, physics_kw, cfg, incumbent, box,
                          free, oracle_target, a): (arm, s)
                for (arm, s) in tasks}
        for fut in as_completed(futs):
            arm, s = futs[fut]
            done += 1
            try:
                log, events, n_false, n_reset = fut.result()
            except Exception as e:            # one bad bout must not kill the run
                failed.append((arm, s, repr(e)))
                print(f"  [{done:3d}/{len(tasks)}] [{arm:7s} seed{s}] FAILED: {e!r}",
                      flush=True)
                continue
            np.savez_compressed(os.path.join(out_logs, f"{arm}_seed{s}.npz"),
                                **{k: v for k, v in log.items()})
            nf = sum(e["fell"] for e in events)
            dist = float(log["y"][-1] - log["y"][0]) if len(log["y"]) else 0.0
            edist = float(sum(e["dist"] for e in events))   # distance under fault
            per[(arm, s)] = dict(events=events, dist=dist, edist=edist,
                                 nf=nf, n_reset=n_reset)
            print(f"  [{done:3d}/{len(tasks)}] [{arm:7s} seed{s}] {nf:2d} falls/bout "
                  f"over {len(events)} events; trial distance {dist:5.1f} m; "
                  f"silent resets {n_reset}", flush=True)
    if failed:
        print(f"\n  !! {len(failed)} bout(s) FAILED: "
              f"{[(m, s) for m, s, _ in failed]}\n", flush=True)

    # ── per-method aggregation (in arm/seed order -> deterministic CSV) ────────
    all_rows = []
    summary = []
    for arm in a.arms:
        seeds_ok = [s for s in seed_ids if (arm, s) in per]
        arm_events = [per[(arm, s)]["events"] for s in seeds_ok]
        arm_dists = [per[(arm, s)]["dist"] for s in seeds_ok]
        arm_edists = [per[(arm, s)]["edist"] for s in seeds_ok]
        arm_falls = [per[(arm, s)]["nf"] for s in seeds_ok]
        for s in seeds_ok:
            for i, e in enumerate(per[(arm, s)]["events"]):
                row = dict(method=arm, seed=s, event=i + 1,
                           onset=e["onset"], detect=e["detect"],
                           latency=e["latency"], fell=e["fell"], V=e["V"],
                           tilt_rms=e["tilt_rms"], dist=e["dist"],
                           false_alarm=int(e["false_alarm"]),
                           request_t=e["request_t"],
                           compute_latency=e["compute_latency"],
                           trial_dist=per[(arm, s)]["dist"])
                for j in free:
                    row[PARAM_NAMES[j]] = float(e["cand"][j])
                all_rows.append(row)
        allev = [e for evs in arm_events for e in evs]
        surv = [e for e in allev if not e["fell"] and e["tilt_rms"] == e["tilt_rms"]]
        tilt = float(np.mean([e["tilt_rms"] for e in surv])) if surv else float("nan")
        fpb = float(np.mean(arm_falls)) if arm_falls else float("nan")
        fpb_sem = float(np.std(arm_falls, ddof=1) / np.sqrt(len(arm_falls))) \
            if len(arm_falls) > 1 else 0.0
        summary.append(dict(method=arm, n_seeds=len(seeds_ok),
                            falls_per_bout=fpb, falls_sem=fpb_sem,
                            mean_tilt_surv=tilt,
                            mean_dist_under_fault=float(np.mean(arm_edists)) if arm_edists else float("nan"),
                            mean_trial_dist=float(np.mean(arm_dists)) if arm_dists else float("nan")))
        print(f"  == {arm:7s}: {fpb:.1f} falls/bout  mean surviving tilt {tilt:.1f} deg  "
              f"dist-under-fault {np.mean(arm_edists) if arm_edists else float('nan'):.1f} m "
              f"(total {np.mean(arm_dists) if arm_dists else float('nan'):.1f} m) ==\n")

    # ── write results for the analysis notebook ──────────────────────────────
    ev_csv = os.path.join(out_results, "continual_events.csv")
    cols = (["method", "seed", "event", "onset", "detect", "latency", "fell",
             "V", "tilt_rms", "dist", "false_alarm", "request_t",
             "compute_latency", "trial_dist"]
            + [PARAM_NAMES[j] for j in free])
    with open(ev_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    sm_csv = os.path.join(out_results, "continual_summary.csv")
    with open(sm_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_seeds", "falls_per_bout",
                                          "falls_sem", "mean_tilt_surv",
                                          "mean_dist_under_fault",
                                          "mean_trial_dist"])
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"saved {ev_csv}\nsaved {sm_csv}\nsaved per-seed traces in {out_logs}")


if __name__ == "__main__":
    main()
