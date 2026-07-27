"""Continual payload-shift adaptation -- all arms in one experiment.

The single, consolidated payload experiment (supersedes the old run_experiment /
run_continual / run_gpsafe trio). One long, non-episodic bout on flat ground:

  * the robot walks normally carrying an 8 kg trunk payload;
  * after the first few seconds the payload SHIFTS off the sagittal plane
    (rearward + lateral, ramped ~1 s) -- a persistent asymmetric CoM offset;
  * a prediction-error CUSUM detects the shift and the chosen METHOD responds;
  * the shift PERSISTS -- it is never auto-reverted. Only a FALL reverts it: the
    robot is stood back upright at its position, the payload is RECENTERED, and
    after a random 2-8 s gap the shift re-engages (the next fall cycle). A method
    that adapts and does not fall keeps walking under the shift for the rest of
    the bout. Search-based methods carry their memory across recurrences.

Headline metric = FALLS PER BOUT: no-adapt tips over repeatedly; a good adapter
falls rarely.

The arms (event_responders.ALL_ARMS, plus `aif` on request), all searching the
same reduced CPG dims (FREE_DIMS_PAYLOAD) so the comparison is head-to-head:

  noadapt -> hold the flat-optimal gait (lower anchor);
  grid    -> Latin-hypercube proposals (naive search);
  bo      -> GP-UCB on the per-event stability score;
  esc     -> extremum-seeking control: sinusoidal-dither, demodulated gradient
             over the reduced dims (model-free classical online tuner);
  safegp  -> the safe GP recovery agent (methods.gp_safe_agent);
  oracle  -> jump to the pre-fit post-shift optimum (upper anchor);
  aif     -> the unified active-inference agent (methods.aif_recovery): a MARX
             model drives its OWN in-process CUSUM trigger while a GP picks the
             gait; not in ALL_ARMS, request it by name.

Per event (each ends at a fall, or at the bout end for a surviving gait) we
record fall / stability (RMS body tilt) / distance; results are written for the
analysis notebook (analyze.ipynb) to load:
  results/continual_events.csv   one row per event, tagged with `method`/`seed`
  results/continual_summary.csv  per-method aggregates (falls/bout, tilt, distance)
  results/logs/<method>_seed<k>.npz   per-seed step traces (incl. cumulative falls)

Usage (from repo root):
    python experiment-simulation/experiment-payload-adapt/run_experiment.py --seeds 5 --duration 120
    python experiment-simulation/experiment-payload-adapt/run_experiment.py --arms noadapt safegp oracle
    # oracle arm needs results/payload_optima.json (python fit_payload_oracles.py)
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
# experiment-simulation/experiment-payload-adapt/ -> repo root (two levels up)
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from methods import continual_driver as cd
from methods import event_responders as er
from methods import responder_worker as rw

RESULTS_DIR = os.path.join(_HERE, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
OPTIMA_JSON = os.path.join(RESULTS_DIR, "payload_optima.json")
INCUMBENT_JSON = os.path.join(RESULTS_DIR, "incumbent.json")

PARAM_NAMES = ["coupling", "w_swing", "w_stance", "F_FAST", "STOP", "hipA", "kneeA", "b"]

# The flat-optimal incumbent (from the now-archived experiment-flat BO fit). Kept
# here so the folder is self-contained; overridable via results/incumbent.json.
INCUMBENT = np.array([7.607, 13.0498, 25.0, 52.4044, 0.5, 0.1, 0.5, 10.0])

# CPG dims the centered<->shifted optimum actually moves (from the oracle refit):
# coupling, w_swing, F_FAST, STOP, b differ; w_stance/hip/knee barely.
FREE_DIMS_PAYLOAD = [0, 1, 3, 4, 7]

# ── Payload scenario defaults (screened 2026-07; see README) ─────────────────
DT = 0.01
LEG_NAMES = ["FL", "FR", "RL", "RR"]
DEFAULT_ORI = [0.0, 0.5, 0.5, 0.0]
PAYLOAD_MASS = 8.0         # payload mass [kg] (trunk link is 10 kg)
PAYLOAD_UP = 0.15          # payload height above trunk center at attach [m]
SHIFT_LAT = 0.215          # world +X (lateral) shift when engaged [m]
SHIFT_BACK = 0.20          # world -Y (rearward) shift when engaged [m]
SHIFT_RAMP_T = 1.0         # shift is ramped over this long [s] (no impulse)
CONSTRAINT_FORCE = 2000.0  # fixed-constraint maxForce [N]
SETTLE_STEPS = 60          # upright-reset settle [steps]
# The shift persists until a fall; EVAL_HOLD is only the trailing window (s) over
# which a SURVIVING gait's stability score V is measured for the responder's
# memory -- NOT an auto-revert timer.
EVAL_HOLD = 10.0

UPRIGHT_FALL = 0.30        # world-up . body-up below which = tip-over
HEIGHT_FALL = 0.25         # base height below which = collapsed [m] (flat ground)


def load_incumbent():
    if os.path.exists(INCUMBENT_JSON):
        return np.asarray(json.load(open(INCUMBENT_JSON))["params"], float)
    return INCUMBENT.copy()


def load_oracle_target():
    """Post-shift optimum (oracle arm); None if not yet fit."""
    if not os.path.exists(OPTIMA_JSON):
        return None
    return np.asarray(json.load(open(OPTIMA_JSON))["shifted"]["params"], float)


def shift_pivot(lat, back, frac):
    """Child-frame pivot realising a payload displacement of frac*(lat,-back) in
    the horizontal plane (robot walks +Y, so `back` moves the load rearward)."""
    return [-frac * float(lat), frac * float(back), 0.0]


class PayloadPhysics:
    """Flat ground + a trunk payload whose CoM shifts off the sagittal plane at
    event intensity `frac` (0 centered .. 1 fully shifted). Implements the
    continual_driver physics contract."""

    def __init__(self, mass=PAYLOAD_MASS, lat=SHIFT_LAT, back=SHIFT_BACK,
                 up=PAYLOAD_UP):
        self.mass, self.lat, self.back, self.up = mass, lat, back, up
        self.p = None

    # ── physics contract ─────────────────────────────────────────────────────
    def setup(self, seed):
        import pybullet as p
        from methods import terrain
        from methods.marxefe_optimizer import (load_environment, load_robot,
                                               JointCPG)
        self._p = p
        JointCPG.ATTITUDE_FEEDBACK = True
        terrain.TERRAIN_CONFIG = {"kind": "flat"}
        load_environment(DT, use_gui=False)
        self.robot, _, self.jids, _, self.feet = load_robot(p)
        self.payload, self.cid = self._attach_payload(frac0=0.0)
        return self._settle([0.0, 0.0], seed)

    def actuate(self, cpg, applied, roll, pitch, frac):
        p = self._p
        # apply the payload shift at intensity frac
        p.changeConstraint(self.cid,
                           jointChildPivot=shift_pivot(self.lat, self.back, frac),
                           maxForce=CONSTRAINT_FORCE)
        raw = np.array([int(len(p.getContactPoints(
            bodyA=0, bodyB=self.robot, linkIndexA=-1, linkIndexB=self.feet[j])) > 0)
            for j in range(4)])
        hips, knees = cpg.step(applied, raw, DT, roll=roll, pitch=pitch)
        for j in range(4):
            a_id, h_id, k_id = self.jids[LEG_NAMES[j]]
            p.setJointMotorControl2(self.robot, a_id, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500)
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

    def _attach_payload(self, frac0=0.0):
        """Rigidly attach a payload box `up` m above the trunk (mass only, no
        collisions), displaced by frac0*(lat,-back). Returns (body_id, cid)."""
        p = self._p
        half = 0.06
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
        anchor_off = np.array([0.0, 0.0, float(self.up)])
        child_pivot = shift_pivot(self.lat, self.back, frac0)
        payload = p.createMultiBody(
            baseMass=float(self.mass), baseCollisionShapeIndex=col,
            basePosition=(np.asarray(base_pos) + anchor_off
                          - np.asarray(child_pivot)).tolist())
        p.setCollisionFilterGroupMask(payload, -1, 0, 0)   # mass only, no contacts
        rot = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
        parent_pivot = rot.T @ anchor_off
        cid = p.createConstraint(self.robot, -1, payload, -1, p.JOINT_FIXED,
                                 [0, 0, 0], parent_pivot.tolist(), child_pivot)
        p.changeConstraint(cid, maxForce=CONSTRAINT_FORCE)
        return payload, cid

    def _recenter_payload(self, at_xy):
        p = self._p
        p.changeConstraint(self.cid, jointChildPivot=shift_pivot(self.lat, self.back, 0.0),
                           maxForce=CONSTRAINT_FORCE)
        p.resetBasePositionAndOrientation(
            self.payload, [at_xy[0], at_xy[1], 0.55 + self.up], [0, 0, 0, 1])
        p.resetBaseVelocity(self.payload, [0, 0, 0], [0, 0, 0])

    def _settle(self, at_xy, seed):
        """Stand the robot upright at (x,y), recenter the payload; fresh CPG."""
        p = self._p
        from methods.marxefe_optimizer import JointCPG
        rng = np.random.default_rng(10_000 + int(seed))
        jit = rng.normal(0.0, 0.002, size=12)
        p.resetBasePositionAndOrientation(self.robot, [at_xy[0], at_xy[1], 0.55],
                                          DEFAULT_ORI)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        self._recenter_payload(at_xy)
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
                                        targetPosition=0.0, force=500)
            for j in hip:
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, 0.25)
            for j in kn:
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, -1.0)
            p.stepSimulation()
        return JointCPG(n_legs=4)


def build_spec(arm, incumbent, box, free, oracle_target, seed, args):
    """Picklable ResponderSpec for the out-of-process responder worker. The
    incumbent is pre-seeded as a known post-shift FALL (bo/safegp) inside the
    worker, so the search-based arms start already knowing not to sit at it."""
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
    physics = PayloadPhysics(**physics_kw)
    if arm == "aif":                       # unified AIF agent: in-process, own trigger
        from methods.aif_recovery import UnifiedAIFAgent
        from methods.continual_driver_aif import run_event_bout_aif
        # decoupled goal: CONTROL keeps tight vx (drives forward), the TRIGGER
        # loosens vx (args.aif_trigger_vx_std) so its cross-entropy is stability-
        # dominated and quiets when the robot is stable, however slow.
        goal_std = (0.25, 0.25, np.deg2rad(12), np.deg2rad(12))   # control (EFE)
        agent = UnifiedAIFAgent(incumbent, box, free, seed, dt=cfg.dt,
                                target_vx=cfg.target_vx, goal_std=goal_std,
                                trigger_vx_std=args.aif_trigger_vx_std)
        return run_event_bout_aif(seed, agent, physics, incumbent, cfg)
    spec = build_spec(arm, incumbent, box, free, oracle_target, seed, args)
    return cd.run_event_bout(seed, spec, physics, incumbent, cfg)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=er.ALL_ARMS,
                    choices=er.ALL_ARMS + ["aif"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--eval-hold", type=float, default=EVAL_HOLD,
                    help="trailing window [s] for scoring a SURVIVING gait's V "
                         "(memory feedback); NOT an auto-revert timer -- the shift "
                         "persists until a fall regardless")
    ap.add_argument("--mass", type=float, default=PAYLOAD_MASS, help="payload mass [kg]")
    ap.add_argument("--shift-lat", type=float, default=SHIFT_LAT,
                    help="lateral shift [m]")
    ap.add_argument("--shift-back", type=float, default=SHIFT_BACK,
                    help="rearward shift [m] (negative = forward)")
    ap.add_argument("--shift-ramp-t", type=float, default=SHIFT_RAMP_T,
                    help="shift ramp duration [s] (shift speed): small = fast/"
                         "impulsive engagement, large = slow/gradual")
    ap.add_argument("--free-dims", type=int, nargs="+", default=None,
                    help=f"CPG dims to search (default {FREE_DIMS_PAYLOAD})")
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
                    help="idealised: react at shift onset instead of detecting")
    ap.add_argument("--detect-kappa", type=float, default=0.20,
                    help="CUSUM slack (tolerated per-step prediction error)")
    ap.add_argument("--detect-h", type=float, default=1.8,
                    help="CUSUM decision threshold (fire when accumulator > h)")
    ap.add_argument("--detect-tau", type=float, default=0.4,
                    help="smoothing time constant for the health signal [s]")
    ap.add_argument("--gap-min", type=float, default=2.0)
    ap.add_argument("--gap-max", type=float, default=8.0)
    ap.add_argument("--grace", type=float, default=2.5,
                    help="post-recenter grace before re-arming the detector [s]")
    ap.add_argument("--aif-trigger-vx-std", type=float, default=1000.0,
                    help="AIF TRIGGER goal-prior std on forward velocity [m/s]. "
                         "Large (default) makes the cross-entropy trigger stability-"
                         "only, so it quiets whenever the robot is stable however "
                         "slow; the CONTROL/EFE goal keeps a tight vx and still "
                         "drives forward. Set ~0.25 to re-couple trigger and control")
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
    ap.add_argument("--out-dir", default=None,
                    help="output directory for CSVs + per-seed logs (default "
                         "results/); inputs (oracle/incumbent JSON) still read "
                         "from results/. Use a separate dir for ablation runs so "
                         "the main results are not overwritten")
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
        raise SystemExit("oracle arm needs results/payload_optima.json "
                         "(run: python experiment-simulation/experiment-payload-adapt/fit_payload_oracles.py)")
    from methods.cpg_bounds import bounds_lower, bounds_upper
    box = (bounds_lower.numpy(), bounds_upper.numpy())
    free = a.free_dims if a.free_dims is not None else FREE_DIMS_PAYLOAD
    physics_kw = dict(mass=a.mass, lat=a.shift_lat, back=a.shift_back)
    cfg = cd.BoutConfig(dt=DT, duration=a.duration, gap_min=a.gap_min,
                        gap_max=a.gap_max, event_ramp_t=a.shift_ramp_t,
                        eval_hold=a.eval_hold,
                        use_detector=not a.no_detector, detect_tau=a.detect_tau,
                        detect_kappa=a.detect_kappa, detect_h=a.detect_h,
                        grace=a.grace, sim_speed=a.sim_speed)

    print(f"continual payload-shift adaptation: arms={a.arms} x {a.seeds} seeds "
          f"x {a.duration:g}s; {a.mass:g}kg shift ({a.shift_lat:g} lat, "
          f"{a.shift_back:g} back), "
          f"persists until a fall; re-engages {a.gap_min:g}-{a.gap_max:g}s after each")
    print(f"  incumbent: {np.round(incumbent, 3).tolist()}")
    print(f"  searching dims {free} ({[PARAM_NAMES[i] for i in free]})")
    det = ("idealised (react at onset)" if a.no_detector else
           f"prediction-error CUSUM (kappa={a.detect_kappa}, h={a.detect_h})")
    print(f"  detector: {det}\n")

    # ── run all (arm, seed) bouts in parallel (each is real-time paced) ───────
    njobs = a.jobs if (a.jobs and a.jobs > 0) else max(1, (os.cpu_count() or 4) - 2)
    tasks = [(arm, s) for arm in a.arms for s in range(a.seeds)]
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
            edist = float(sum(e["dist"] for e in events))   # distance under shift
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
        seeds_ok = [s for s in range(a.seeds) if (arm, s) in per]
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
        # per-method aggregate (real events only, i.e. not false alarms)
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
