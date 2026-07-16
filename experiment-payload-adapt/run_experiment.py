"""Continual payload-shift adaptation -- all five arms in one experiment.

The single, consolidated payload experiment (supersedes the old run_experiment /
run_continual / run_gpsafe trio). One long, non-episodic bout on flat ground:

  * the robot walks normally carrying an 8 kg trunk payload;
  * after the first few seconds the payload SHIFTS off the sagittal plane
    (rearward + lateral, ramped ~1 s) -- a persistent asymmetric CoM offset;
  * a prediction-error CUSUM detects the shift and the chosen METHOD responds;
  * the response is held under the full shift for a few seconds and scored;
  * on a FALL the robot is stood back upright at its current position and the
    payload is RECENTERED (event reverted); either way, after a random 2-8 s the
    shift recurs. Search-based methods carry their memory across recurrences.

Five arms (event_responders.ALL_ARMS), all searching the same reduced CPG
dims (FREE_DIMS_PAYLOAD) so the comparison is head-to-head:

  noadapt -> hold the flat-optimal gait (lower anchor);
  grid    -> Latin-hypercube proposals (naive search);
  bo      -> GP-UCB on the per-event stability score;
  safegp  -> the safe GP recovery agent (methods.gp_safe_agent);
  oracle  -> jump to the pre-fit post-shift optimum (upper anchor).

Per event we record fall / stability (RMS body tilt + composite score V) /
distance travelled; results are written for the analysis notebook to load:
  results/continual_events.csv   one row per event, tagged with `method`/`seed`
  results/continual_summary.csv  per-method aggregates (fall rate, tilt, distance)
  results/logs/<method>_seed<k>.npz   per-seed step traces (for time-series plots)

Usage (from repo root):
    python experiment-payload-adapt/run_experiment.py --seeds 5 --duration 120
    python experiment-payload-adapt/run_experiment.py --arms noadapt safegp oracle
    # oracle arm needs results/payload_optima.json (python fit_payload_oracles.py)
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from methods import continual_driver as cd
from methods import event_responders as er

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
        return cd.StepState(base_pos=base_pos, vx=vel[1], roll=roll, pitch=pitch,
                            fell=fell)

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


def build_responder(arm, incumbent, box, free, oracle_target, seed, args):
    safegp_kwargs = dict(n_init=args.n_init, safe_V=args.safe_V, beta=args.beta,
                         kappa=args.kappa, objective=args.objective,
                         efe_y_star=args.efe_y_star, efe_tau2=args.efe_tau2,
                         efe_adaptive=args.efe_adaptive,
                         efe_tau2_min=args.efe_tau2_min,
                         efe_tau2_max=args.efe_tau2_max)
    r = er.make_responder(arm, incumbent, box, free, oracle_target, seed,
                          safegp_kwargs=safegp_kwargs)
    # seed the search memory: the incumbent is a known post-shift FALL, so the
    # search-based arms start already knowing not to sit at it.
    if arm in ("bo", "safegp"):
        r.update(np.asarray(incumbent, float), cd.BoutConfig().v_fall, True)
    return r


def run_seed(seed, arm, physics_kw, cfg, incumbent, box, free, oracle_target, args):
    physics = PayloadPhysics(**physics_kw)
    responder = build_responder(arm, incumbent, box, free, oracle_target, seed, args)
    return cd.run_event_bout(seed, responder, physics, incumbent, cfg)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=er.ALL_ARMS, choices=er.ALL_ARMS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--mass", type=float, default=PAYLOAD_MASS, help="payload mass [kg]")
    ap.add_argument("--lat", type=float, default=SHIFT_LAT, help="lateral shift [m]")
    ap.add_argument("--back", type=float, default=SHIFT_BACK, help="rearward shift [m]")
    ap.add_argument("--free-dims", type=int, nargs="+", default=None,
                    help=f"CPG dims to search (default {FREE_DIMS_PAYLOAD})")
    # safegp / bo agent knobs
    ap.add_argument("--n-init", type=int, default=4)
    ap.add_argument("--safe-V", type=float, default=-0.8)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--objective", choices=["ucb", "efe"], default="efe",
                    help="safegp planning objective: GP-UCB or Expected Free Energy")
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
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    incumbent = load_incumbent()
    oracle_target = load_oracle_target()
    if "oracle" in a.arms and oracle_target is None:
        raise SystemExit("oracle arm needs results/payload_optima.json "
                         "(run: python experiment-payload-adapt/fit_payload_oracles.py)")
    from methods.cpg_bounds import bounds_lower, bounds_upper
    box = (bounds_lower.numpy(), bounds_upper.numpy())
    free = a.free_dims if a.free_dims is not None else FREE_DIMS_PAYLOAD
    physics_kw = dict(mass=a.mass, lat=a.lat, back=a.back)
    cfg = cd.BoutConfig(dt=DT, duration=a.duration, gap_min=a.gap_min,
                        gap_max=a.gap_max, event_ramp_t=SHIFT_RAMP_T,
                        use_detector=not a.no_detector, detect_tau=a.detect_tau,
                        detect_kappa=a.detect_kappa, detect_h=a.detect_h,
                        grace=a.grace)

    print(f"continual payload-shift adaptation: arms={a.arms} x {a.seeds} seeds "
          f"x {a.duration:g}s; {a.mass:g}kg shift ({a.lat:g} lat, {a.back:g} back) "
          f"recurring every {a.gap_min:g}-{a.gap_max:g}s")
    print(f"  incumbent: {np.round(incumbent, 3).tolist()}")
    print(f"  searching dims {free} ({[PARAM_NAMES[i] for i in free]})")
    det = ("idealised (react at onset)" if a.no_detector else
           f"prediction-error CUSUM (kappa={a.detect_kappa}, h={a.detect_h})")
    print(f"  detector: {det}\n")

    all_rows = []
    summary = []
    for arm in a.arms:
        arm_events = []
        arm_dists = []
        for s in range(a.seeds):
            log, events, n_false, n_reset = run_seed(
                s, arm, physics_kw, cfg, incumbent, box, free, oracle_target, a)
            arm_events.append(events)
            np.savez_compressed(os.path.join(LOG_DIR, f"{arm}_seed{s}.npz"),
                                **{k: v for k, v in log.items()})
            n = len(events)
            nf = sum(e["fell"] for e in events)
            dist = float(log["y"][-1] - log["y"][0]) if len(log["y"]) else 0.0
            arm_dists.append(dist)
            print(f"  [{arm:7s} seed{s}] {n:2d} events, {nf} falls "
                  f"({nf/max(n,1):.0%}); trial distance {dist:5.1f} m; "
                  f"false alarms {n_false}; silent resets {n_reset}", flush=True)
            for i, e in enumerate(events):
                row = dict(method=arm, seed=s, event=i + 1,
                           onset=e["onset"], detect=e["detect"],
                           latency=e["latency"], fell=e["fell"], V=e["V"],
                           tilt_rms=e["tilt_rms"], dist=e["dist"],
                           false_alarm=int(e["false_alarm"]),
                           trial_dist=dist)
                for j in free:
                    row[PARAM_NAMES[j]] = float(e["cand"][j])
                all_rows.append(row)
        # per-method aggregate (real events only, i.e. not false alarms)
        real = [e for evs in arm_events for e in evs if not e["false_alarm"]]
        n = len(real)
        fr = float(np.mean([e["fell"] for e in real])) if n else float("nan")
        surv = [e for e in real if not e["fell"]]
        tilt = float(np.mean([e["tilt_rms"] for e in surv
                              if e["tilt_rms"] == e["tilt_rms"]])) if surv else float("nan")
        meanV = float(np.mean([e["V"] for e in real])) if n else float("nan")
        summary.append(dict(method=arm, n_events=n, fall_rate=fr,
                            mean_tilt_surv=tilt, mean_V=meanV,
                            mean_trial_dist=float(np.mean(arm_dists))))
        print(f"  == {arm:7s}: fall rate {fr:.0%}  mean surviving tilt {tilt:.1f} deg  "
              f"mean V {meanV:+.2f}  mean distance {np.mean(arm_dists):.1f} m ==\n")

    # ── write results for the analysis notebook ──────────────────────────────
    ev_csv = os.path.join(RESULTS_DIR, "continual_events.csv")
    cols = (["method", "seed", "event", "onset", "detect", "latency", "fell",
             "V", "tilt_rms", "dist", "false_alarm", "trial_dist"]
            + [PARAM_NAMES[j] for j in free])
    with open(ev_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    sm_csv = os.path.join(RESULTS_DIR, "continual_summary.csv")
    with open(sm_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_events", "fall_rate",
                                          "mean_tilt_surv", "mean_V",
                                          "mean_trial_dist"])
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"saved {ev_csv}\nsaved {sm_csv}\nsaved per-seed traces in {LOG_DIR}")


if __name__ == "__main__":
    main()
