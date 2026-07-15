"""Continual (non-episodic) leg-damage recovery with the GP-safe agent.

One long continuous bout. The robot walks; at random intervals the SAME hind leg
(RR) is damaged (hip+knee maxForce ramps 60->22 Nm). The moment damage sets in,
the agent reacts by proposing a CPG gait from its GP memory over the control
space (safe acquisition that avoids the parameter regions it remembers falling).
Each damage event is one query, folded straight back into the memory:

  * if the robot FALLS -> record (gait, V_FALL); we stand it back upright and HEAL
    the leg; after a random 2-8 s of healthy walking the damage recurs;
  * if it SURVIVES a hold window under full damage -> record (gait, V); heal, and
    the damage recurs after the same random delay.

So the memory persists across recurrences WITHIN the single trial: the agent
should stop re-trying gaits it has already seen fall and converge onto a
recovery gait. No gait is ever supplied. (Detection is idealised: the agent
reacts at damage onset -- earlier sweeps showed detection latency is not the
bottleneck. The search space is reduced to the high-leverage CPG dims.)

Usage (from repo root):
    python experiment-damage-adapt/run_continual.py --duration 120 --seeds 3
"""

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
PARAM_NAMES = ["coupling", "w_swing", "w_stance", "F_FAST", "STOP", "hipA", "kneeA", "b"]

# event/scheduling constants
DAMAGE_RAMP_T = 0.8       # leg torque droop ramp [s]
PARAM_RAMP = 30           # steps to ramp a gait switch in (0.3 s)
EVAL_HOLD = 3.0           # survive this long at full damage -> success [s]
GAP_MIN, GAP_MAX = 3.0, 8.0   # random healthy interval before damage recurs [s]
FIRST_DAMAGE_T = 4.0      # first damage onset [s]
SETTLE_STEPS = 60         # upright-reset settle [steps]
TARGET_VX = 0.5


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _score(vx, roll, pitch):
    """Post-damage V over an event window (same scale as the GP memory)."""
    if len(vx) < 20:
        return -2.0
    r_v = min(max(float(np.mean(vx)), 0.0) / TARGET_VX, 1.0)
    rms_roll = np.rad2deg(np.sqrt(np.mean(np.asarray(roll) ** 2)))
    rms_pitch = np.rad2deg(np.sqrt(np.mean(np.asarray(pitch) ** 2)))
    return float(r_v - (rms_roll + rms_pitch) / 10.0)


def run_seed(seed, duration, dm, gp, free_dims, n_init, args):
    import pybullet as p
    from methods import terrain
    from methods.marxefe_optimizer import (get_base_orientation,
                                           load_environment, load_robot, JointCPG)
    DT = dm.DT
    LEG = dm.LEG_NAMES
    ORI = dm.DEFAULT_ORI
    HF, DF, ABD = dm.HEALTHY_FORCE, dm.DAMAGE_FORCE, dm.ABD_FORCE
    incumbent = dm.load_incumbent()
    from methods.cpg_bounds import bounds_lower, bounds_upper
    box = (bounds_lower.numpy(), bounds_upper.numpy())

    JointCPG.ATTITUDE_FEEDBACK = True
    terrain.TERRAIN_CONFIG = {"kind": "flat"}
    load_environment(DT, use_gui=False)
    robot, _, jids, _, feet = load_robot(p)
    dmg_j = LEG.index(dm.DAMAGE_LEG)
    dmg_hk = jids[LEG[dmg_j]][1:3]

    def settle(at_xy, s):
        rng = np.random.default_rng(10_000 + int(s))
        jit = rng.normal(0.0, 0.002, size=12)
        p.resetBasePositionAndOrientation(robot, [at_xy[0], at_xy[1], 0.55], ORI)
        p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0])
        abd, hip, kn = [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14]
        for i, j in enumerate(abd):
            p.resetJointState(robot, j, 0.0 + jit[i])
        for i, j in enumerate(hip):
            p.resetJointState(robot, j, 0.05 + jit[4 + i])
        for i, j in enumerate(kn):
            p.resetJointState(robot, j, -0.6 + jit[8 + i])
        for _ in range(SETTLE_STEPS):
            for j in abd:
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=0.0, force=ABD)
            for j in hip:
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, 0.25)
            for j in kn:
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, -1.0)
            p.stepSimulation()
        return JointCPG(n_legs=4)

    cpg = settle([0.0, 0.0], seed)
    agent = gp.GPSafeRecovery(incumbent, box, free_dims=free_dims, seed=seed,
                              n_init=n_init, safe_V=args.safe_V, beta=args.beta,
                              kappa=args.kappa, objective=args.objective,
                              efe_y_star=args.efe_y_star, efe_tau2=args.efe_tau2,
                              efe_adaptive=args.efe_adaptive,
                              efe_tau2_min=args.efe_tau2_min,
                              efe_tau2_max=args.efe_tau2_max, archive_path=None)
    agent.update(incumbent, -2.0, True)          # seed: incumbent falls under damage
    rng = np.random.default_rng(777 + seed)

    n_steps = int(round(duration / DT))
    log = {k: np.zeros(n_steps) for k in
           ("t", "y", "vx", "roll", "pitch", "legf", "state", "dmg", "cusum")}
    events = []

    # ── prediction-error detector (CUSUM on a smoothed health signal) ────────
    # The agent does NOT know when the leg is damaged; it watches its own
    # prediction error -- the forward-speed deficit + extra tilt relative to the
    # healthy baseline -- and fires when the accumulated error crosses H. The
    # detector re-arms after each heal (with a grace period while the robot
    # re-stabilises). `damage_active` (physical) is decoupled from detection.
    use_det = not args.no_detector
    a_s = DT / max(args.detect_tau, DT)          # EMA smoothing coefficient
    TIP_SCALE = np.deg2rad(12.0)
    vx_s = tip_s = None                          # smoothed signals
    S = 0.0                                       # CUSUM statistic
    vx_base = tip_base = None                     # healthy reference (set after warmup)
    warm_vx, warm_tip = [], []

    phase = "monitoring"                          # or "responding"
    damage_active = False
    onset_t = None                                # physical damage onset
    detect_t = None                               # when the agent noticed
    armed = False                                 # detector armed (health-gated)
    heal_streak = 0                               # consecutive healthy steps
    ARM_STREAK = int(round(0.5 / DT))             # require 0.5 s healthy before arming
    next_damage_t = FIRST_DAMAGE_T
    grace_until = 1.5                             # earliest re-arm after a heal
    cand = incumbent.copy()
    seg_start = incumbent.copy(); seg_target = incumbent.copy(); seg_anchor = 0
    win = {"vx": [], "roll": [], "pitch": []}
    roll = pitch = 0.0
    n_false = 0
    n_reset = 0                                   # silent recoveries (healthy falls)

    def _resolve(fell, V, base_pos, k, t):
        """Record the outcome, heal the leg, (reset upright if fell), disarm."""
        nonlocal phase, damage_active, next_damage_t, grace_until, cpg
        nonlocal seg_start, seg_target, seg_anchor, win, roll, pitch, S
        nonlocal onset_t, detect_t, armed
        agent.update(cand, V, fell)
        events.append(dict(onset=onset_t, detect=detect_t,
                           latency=(detect_t - onset_t if (detect_t is not None
                                    and onset_t is not None) else np.nan),
                           fell=int(fell), V=float(V), cand=cand.copy(),
                           false_alarm=(onset_t is None)))
        if fell:
            cpg = settle([base_pos[0], base_pos[1]], seed * 131 + len(events))
            roll = pitch = 0.0
            seg_start = incumbent.copy()
        else:
            seg_start = None                       # filled below from `applied`
        seg_target = incumbent.copy()
        seg_anchor = k + 1
        win = {"vx": [], "roll": [], "pitch": []}
        damage_active = False
        onset_t = None; detect_t = None            # clear so a stray fire = false alarm
        armed = False                              # re-arm only after recovery
        phase = "monitoring"
        next_damage_t = t + rng.uniform(GAP_MIN, GAP_MAX)
        grace_until = t + args.grace
        S = 0.0

    k = 0
    while k < n_steps:
        t = k * DT
        # physical damage schedule (agent is blind to this)
        if phase == "monitoring" and not damage_active and t >= next_damage_t:
            damage_active = True; onset_t = t; detect_t = None
        legf = (HF + min(1.0, (t - onset_t) / DAMAGE_RAMP_T) * (DF - HF)
                if damage_active else HF)

        frac_p = min(1.0, (k - seg_anchor) / max(1, PARAM_RAMP))
        if seg_start is None:                      # survived-heal: ramp from current
            seg_start = applied.copy()
        applied = seg_start + frac_p * (seg_target - seg_start)

        raw = np.array([int(len(p.getContactPoints(bodyA=0, bodyB=robot,
              linkIndexA=-1, linkIndexB=feet[j])) > 0) for j in range(4)])
        hips, knees = cpg.step(applied, raw, DT, roll=roll, pitch=pitch)
        for j in range(4):
            a_id, h_id, k_id = jids[LEG[j]]
            p.setJointMotorControl2(robot, a_id, p.POSITION_CONTROL, targetPosition=0.0, force=ABD)
            if j == dmg_j:
                p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j], force=legf)
                p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j], force=legf)
            else:
                p.setJointMotorControl2(robot, h_id, p.POSITION_CONTROL, hips[j])
                p.setJointMotorControl2(robot, k_id, p.POSITION_CONTROL, knees[j])
        p.stepSimulation()

        base_pos, base_ori = get_base_orientation(p, robot, ORI)
        vel, _ = p.getBaseVelocity(robot)
        pitch, roll, _ = p.getEulerFromQuaternion(base_ori)
        fell, _up = dm._fallen_flat(base_pos, base_ori, p)
        vx = vel[1]; tipmag = np.hypot(roll, pitch)

        # smoothed signals + healthy baseline (measured during warmup window)
        vx_s = vx if vx_s is None else vx_s + a_s * (vx - vx_s)
        tip_s = tipmag if tip_s is None else tip_s + a_s * (tipmag - tip_s)
        if vx_base is None:
            if 1.5 <= t < FIRST_DAMAGE_T:
                warm_vx.append(vx); warm_tip.append(tipmag)
            if t >= FIRST_DAMAGE_T - DT:
                vx_base = max(float(np.mean(warm_vx)), 0.1) if warm_vx else 0.5
                tip_base = float(np.mean(warm_tip)) if warm_tip else 0.0

        # Arm the detector only once the robot has been walking at healthy speed
        # for a SUSTAINED window (past the grace period), so the post-heal recovery
        # transient is never mistaken for damage. Then run the CUSUM on the
        # prediction error.
        healthy_now = (vx_base is not None and phase == "monitoring"
                       and t >= grace_until and vx_s >= 0.85 * vx_base
                       and tip_s <= tip_base + np.deg2rad(4.0))
        heal_streak = heal_streak + 1 if healthy_now else 0
        if not armed and heal_streak >= ARM_STREAK:
            armed = True; S = 0.0
        e = 0.0
        if armed and phase == "monitoring":
            e = (max(0.0, vx_base - vx_s) / vx_base
                 + max(0.0, tip_s - tip_base) / TIP_SCALE)
            S = max(0.0, S + e - args.detect_kappa)
        log["t"][k], log["y"][k] = t, base_pos[1]
        log["vx"][k], log["roll"][k], log["pitch"][k] = vx, roll, pitch
        log["legf"][k] = legf
        log["state"][k] = 1.0 if phase == "responding" else 0.0
        log["dmg"][k] = 1.0 if damage_active else 0.0
        log["cusum"][k] = S

        if phase == "responding":
            if t - detect_t > PARAM_RAMP * DT + 0.2:     # collect after the ramp-in
                win["vx"].append(vx); win["roll"].append(roll); win["pitch"].append(pitch)
            if fell:
                _resolve(True, -2.0, base_pos, k, t)
            elif t - detect_t >= PARAM_RAMP * DT + EVAL_HOLD:
                _resolve(False, _score(win["vx"], win["roll"], win["pitch"]),
                         base_pos, k, t)
        else:  # monitoring
            fired = (armed and S > args.detect_h) if use_det \
                else (damage_active and detect_t is None)
            if fell and onset_t is not None:             # undetected DAMAGE fall
                cand = incumbent.copy()                   # robot was running incumbent
                _resolve(True, -2.0, base_pos, k, t)
            elif fell:                                   # residual fall while HEALTHY
                # (leftover instability from the prior gait / reset) -- not a
                # damage response: recover silently, no event, no memory update.
                cpg = settle([base_pos[0], base_pos[1]], seed * 131 + 90000 + k)
                roll = pitch = 0.0
                seg_start = incumbent.copy(); seg_target = incumbent.copy()
                seg_anchor = k + 1
                armed = False; heal_streak = 0; S = 0.0
                grace_until = t + args.grace
                n_reset += 1
            elif fired:
                detect_t = t
                if onset_t is None:
                    n_false += 1                         # fired with no active damage
                cand, _mode = agent.propose()
                seg_start = applied.copy(); seg_target = cand.copy(); seg_anchor = k + 1
                win = {"vx": [], "roll": [], "pitch": []}
                phase = "responding"; armed = False; S = 0.0
        k += 1

    p.disconnect()
    for key in log:
        log[key] = log[key][:k]
    return log, events, agent, n_false, n_reset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--free-dims", type=int, nargs="+", default=None)
    ap.add_argument("--n-init", type=int, default=4)
    ap.add_argument("--safe-V", type=float, default=-0.8)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--objective", choices=["ucb", "efe"], default="efe",
                    help="agent planning objective: GP-UCB or Expected Free Energy")
    ap.add_argument("--efe-y-star", type=float, default=1.0,
                    help="EFE preferred outcome y* (optimistic post-damage V)")
    ap.add_argument("--efe-tau2", type=float, default=0.5,
                    help="EFE preference variance tau^2 (fixed); low=exploit, high=explore")
    ap.add_argument("--efe-adaptive", action="store_true",
                    help="curvature-aware tau^2 (Anil Meera & Kouw eq. 11-12)")
    ap.add_argument("--efe-tau2-min", type=float, default=0.1)
    ap.add_argument("--efe-tau2-max", type=float, default=3.0)
    ap.add_argument("--no-detector", action="store_true",
                    help="idealised: react at damage onset instead of detecting "
                         "the prediction error (for comparison)")
    ap.add_argument("--detect-kappa", type=float, default=0.15,
                    help="CUSUM slack (tolerated per-step prediction error)")
    ap.add_argument("--detect-h", type=float, default=1.8,
                    help="CUSUM decision threshold (fire when accumulator > h)")
    ap.add_argument("--detect-tau", type=float, default=0.4,
                    help="smoothing time constant for the health signal [s]")
    ap.add_argument("--grace", type=float, default=1.5,
                    help="post-heal grace before re-arming the detector [s]")
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dm = _load("run_experiment")
    from methods import gp_safe_agent as gp   # moved to methods/ (shared)
    free = a.free_dims if a.free_dims is not None else gp.FREE_DIMS_DEFAULT
    print(f"continual leg-damage recovery: {a.seeds} seeds x {a.duration:g}s, "
          f"RR 60->22 Nm recurring every {GAP_MIN:g}-{GAP_MAX:g}s")
    obj = (f"EFE (y*={a.efe_y_star}, tau2="
           f"{'adaptive[%.1f,%.1f]' % (a.efe_tau2_min, a.efe_tau2_max) if a.efe_adaptive else a.efe_tau2})"
           if a.objective == "efe" else f"GP-UCB (beta={a.beta})")
    det = ("idealised (react at onset)" if a.no_detector else
           f"prediction-error CUSUM (kappa={a.detect_kappa}, h={a.detect_h}, "
           f"tau={a.detect_tau}s)")
    print(f"  objective: {obj}")
    print(f"  detector: {det}")
    print(f"  searching dims {free} ({[PARAM_NAMES[i] for i in free]}); "
          f"heal+upright on fall; hold {EVAL_HOLD:g}s to confirm survival\n")

    all_events = {}
    logs = {}
    n_false_tot = 0
    for s in range(a.seeds):
        log, events, agent, n_false, n_reset = run_seed(s, a.duration, dm, gp, free, a.n_init, a)
        all_events[s] = events
        logs[s] = log
        n_false_tot += n_false
        n = len(events)
        nf = sum(e["fell"] for e in events)
        third = max(1, n // 3)
        fr_early = np.mean([e["fell"] for e in events[:third]])
        fr_late = np.mean([e["fell"] for e in events[-third:]])
        bestV = max([e["V"] for e in events], default=float("nan"))
        lats = [e["latency"] for e in events if e["latency"] == e["latency"]]
        mlat = float(np.mean(lats)) if lats else float("nan")
        print(f"seed {s}: {n} damage events, {nf} falls ({nf/n:.0%}); "
              f"fall rate first third {fr_early:.0%} -> last third {fr_late:.0%}; "
              f"best V={bestV:+.3f}; det.latency={mlat:.2f}s; "
              f"false alarms={n_false}; silent resets={n_reset}; "
              f"final memory {len(agent.Y)} entries")
        for i, e in enumerate(events):
            fd = " ".join(f"{PARAM_NAMES[j]}={e['cand'][j]:.2f}" for j in free)
            tag = "FALSE-ALARM" if e["false_alarm"] else (
                "FELL " if e["fell"] else "walked")
            lat = "" if e["latency"] != e["latency"] else f"lat={e['latency']:.2f}s "
            print(f"    ev{i+1:2d} t={e['onset'] if e['onset'] is not None else -1:5.1f}s "
                  f"{tag} {lat}V={e['V']:+.3f} | {fd}", flush=True)

    # aggregate across seeds
    ev_all = [e for s in range(a.seeds) for e in all_events[s]]
    print(f"\n=== continual summary ({a.seeds} seeds) ===")
    tot = len(ev_all); tf = sum(e["fell"] for e in ev_all)
    lats = [e["latency"] for e in ev_all if e["latency"] == e["latency"]]
    print(f"  total damage events {tot}, falls {tf} ({tf/tot:.0%}); "
          f"mean detection latency {np.mean(lats):.2f}s; false alarms {n_false_tot}")
    fr_e = np.mean([np.mean([e["fell"] for e in all_events[s][:max(1, len(all_events[s])//3)]])
                    for s in range(a.seeds)])
    fr_l = np.mean([np.mean([e["fell"] for e in all_events[s][-max(1, len(all_events[s])//3):]])
                    for s in range(a.seeds)])
    print(f"  mean fall rate: first third {fr_e:.0%} -> last third {fr_l:.0%} "
          f"({'LEARNING (falls drop)' if fr_l < fr_e - 1e-9 else 'no drop'})")

    # csv of all events
    csvp = os.path.join(RESULTS_DIR, "continual_events.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "event", "onset_t", "detect_t", "latency", "fell",
                    "false_alarm", "V"] + [PARAM_NAMES[i] for i in free])
        for s in range(a.seeds):
            for i, e in enumerate(all_events[s]):
                w.writerow([s, i + 1,
                            "" if e["onset"] is None else f"{e['onset']:.2f}",
                            "" if e["detect"] is None else f"{e['detect']:.2f}",
                            "" if e["latency"] != e["latency"] else f"{e['latency']:.3f}",
                            e["fell"], int(e["false_alarm"]), f"{e['V']:.4f}"]
                           + [f"{e['cand'][j]:.4f}" for j in free])
    print(f"\nsaved {csvp}")

    _plot(logs, all_events, a.seeds, free, dm)


def _plot(logs, all_events, seeds, free, dm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(2, 1, figsize=(13, 8))
        # top: seed-0 vx trace; pink band = leg physically damaged, gray band =
        # agent responding; ▽ = physical onset, ▼ = detection, x = fall
        lg = logs[0]
        ax[0].plot(lg["t"], lg["vx"], lw=0.7, color="tab:blue")
        ax[0].fill_between(lg["t"], -0.5, 1.2, where=lg["dmg"] > 0.5,
                           color="crimson", alpha=0.10, lw=0, label="leg damaged")
        ax[0].fill_between(lg["t"], -0.5, 1.2, where=lg["state"] > 0.5,
                           color="tab:green", alpha=0.12, lw=0, label="agent responding")
        for e in all_events[0]:
            if e["onset"] is not None:
                ax[0].plot(e["onset"], 1.12, "v", color="gray", ms=6)      # onset
            if e["detect"] is not None:
                ax[0].plot(e["detect"], 1.12, "v", color="tab:orange", ms=6)  # detect
            if e["fell"]:
                ax[0].plot((e["detect"] or e["onset"]), 1.0, "x", color="red", ms=8)
        ax[0].axhline(0.5, color="k", ls=":", lw=0.8)
        ax[0].set_ylim(-0.5, 1.2); ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("vx [m/s]")
        ax[0].set_title("Seed 0: continual recovery (pink=damaged, green=responding, "
                        "▽ onset, ▼ detect, ✕ fall)")
        ax[0].legend(loc="upper right", fontsize=8, ncol=2); ax[0].grid(alpha=0.3)
        # bottom: per-event V across all seeds vs event index (learning curve)
        cols = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]
        for s in range(seeds):
            ev = all_events[s]
            xs = np.arange(1, len(ev) + 1)
            Vs = [e["V"] for e in ev]
            fell = [e["fell"] for e in ev]
            ax[1].plot(xs, Vs, "-", color=cols[s % len(cols)], alpha=0.5, lw=1)
            ax[1].scatter(xs, Vs, c=["crimson" if f else cols[s % len(cols)] for f in fell],
                          s=26, label=f"seed {s}")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xlabel("damage event # (within trial)")
        ax[1].set_ylabel("post-damage V (red=fell)")
        ax[1].set_title("Per-event outcome over the trial (falls should thin out as memory fills)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "continual_recovery.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out}")
    except Exception as ex:
        print(f"(figure skipped: {ex})")


if __name__ == "__main__":
    main()
