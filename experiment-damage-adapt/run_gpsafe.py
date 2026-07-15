"""Driver for the GP-safe recovery agent (gp_safe_agent.GPSafeRecovery).

Runs repeated leg-damage EPISODES. Each episode the agent proposes one post-
damage gait from its GP memory; the gait is evaluated under the damaged leg
(RR 60->22 Nm, held over the scored tail -- the same scorer that defined the
oracle), and the outcome V (falls -> V_FALL) is folded back into the memory. The
agent is never told the recovery gait; it discovers one by remembering which
control-space regions fall and steering its safe-UCB search away from them.

Demonstrates: (1) does the agent DISCOVER a survivor (V>0) on its own, and by
which episode; (2) does the fall rate over episodes drop as the memory fills;
(3) how the discovered gait compares to the incumbent (falls) and the BO oracle.
The (params, V, fell) memory persists to results/gpsafe_archive.npz across runs.

Runs SEQUENTIALLY in-process so the memory accumulates cleanly across episodes.

Usage (from repo root):
    python experiment-damage-adapt/run_gpsafe.py --episodes 40 --eval-seeds 2
    python experiment-damage-adapt/run_gpsafe.py --free-dims 0 7   # ultra-reduced (coupling,b)
    python experiment-damage-adapt/run_gpsafe.py --fresh           # ignore saved memory
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS_DIR = os.path.join(_HERE, "results")
ARCHIVE = os.path.join(RESULTS_DIR, "gpsafe_archive.npz")
LOG_CSV = os.path.join(RESULTS_DIR, "gpsafe_episodes.csv")
FIG = os.path.join(RESULTS_DIR, "figures", "gpsafe_convergence.png")
PARAM_NAMES = ["coupling", "w_swing", "w_stance", "F_FAST", "STOP", "hipA", "kneeA", "b"]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--eval-seeds", type=int, default=2,
                    help="seeds averaged per candidate (episode outcome robustness)")
    ap.add_argument("--n-init", type=int, default=6)
    ap.add_argument("--free-dims", type=int, nargs="+", default=None,
                    help="CPG param indices to search; rest frozen at incumbent "
                         "(default: 0 1 4 5 7 = coupling w_swing STOP hipA b)")
    ap.add_argument("--safe-V", type=float, default=-0.8)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--objective", choices=["ucb", "efe"], default="ucb",
                    help="agent planning objective: GP-UCB or Expected Free Energy")
    ap.add_argument("--efe-y-star", type=float, default=1.0)
    ap.add_argument("--efe-tau2", type=float, default=0.5)
    ap.add_argument("--efe-adaptive", action="store_true")
    ap.add_argument("--efe-tau2-min", type=float, default=0.1)
    ap.add_argument("--efe-tau2-max", type=float, default=3.0)
    ap.add_argument("--fresh", action="store_true", help="ignore any saved memory")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    from methods.marxefe_optimizer import JointCPG
    JointCPG.ATTITUDE_FEEDBACK = True
    from methods.cpg_bounds import bounds_lower, bounds_upper
    box = (bounds_lower.numpy(), bounds_upper.numpy())
    dm = _load("run_experiment")
    fit = _load("fit_damage_oracles")
    from methods import gp_safe_agent as gp   # moved to methods/ (shared)
    incumbent = dm.load_incumbent()

    # references: incumbent (should fall) and the BO oracle (should walk)
    def evalc(full_x, seeds):
        Vs, falls = [], 0
        for s in range(seeds):
            res = dm.run_trial(s, "noadapt", 1.0, np.asarray(full_x, float), box,
                               fit._cond_cfg(dm, "damaged"), trigger="ce",
                               duration=fit.FIT_DURATION)
            Vs.append(fit._score_tail(res)); falls += int(res["fell"])
        return float(np.mean(Vs)), falls / seeds

    free = a.free_dims if a.free_dims is not None else gp.FREE_DIMS_DEFAULT
    frozen = [i for i in range(8) if i not in free]
    print(f"GP-safe recovery: searching dims {free} "
          f"({[PARAM_NAMES[i] for i in free]}); frozen "
          f"{[PARAM_NAMES[i] for i in frozen]} at incumbent")
    print(f"  eval-seeds={a.eval_seeds}  n_init={a.n_init}  safe_V={a.safe_V}  "
          f"beta={a.beta}  kappa={a.kappa}")

    inc_V, inc_fell = evalc(incumbent, a.eval_seeds)
    print(f"  reference incumbent under damage: V={inc_V:+.3f} fell={inc_fell:.0%}")
    opt = np.asarray(json.load(open(dm.OPTIMA_JSON))["damaged"]["params"], float)
    optred = incumbent.copy(); optred[free] = opt[free]      # oracle projected to the reduced space
    optV, optfell = evalc(optred, a.eval_seeds)
    print(f"  reference oracle (proj. to reduced space): V={optV:+.3f} fell={optfell:.0%} "
          f"-> reduced space {'CONTAINS' if optV > 0 else 'may NOT contain'} a survivor")

    archive_path = None if a.fresh else ARCHIVE
    agent = gp.GPSafeRecovery(incumbent, box, free_dims=free, seed=a.seed,
                              n_init=a.n_init, safe_V=a.safe_V, beta=a.beta,
                              kappa=a.kappa, objective=a.objective,
                              efe_y_star=a.efe_y_star, efe_tau2=a.efe_tau2,
                              efe_adaptive=a.efe_adaptive,
                              efe_tau2_min=a.efe_tau2_min,
                              efe_tau2_max=a.efe_tau2_max,
                              archive_path=archive_path)
    # seed memory with the incumbent's fall (the agent has just experienced it)
    if len(agent.Y) == 0:
        agent.update(incumbent, inc_V, inc_fell > 0.5)
    print(f"  memory starts with {len(agent.Y)} remembered episodes\n")

    rows = []
    first_survivor = None
    for e in range(a.episodes):
        full_x, mode = agent.propose()
        V, fell = evalc(full_x, a.eval_seeds)
        agent.update(full_x, V, fell)
        best_x, best_V = agent.best()
        if fell <= 0.5 and first_survivor is None:      # first non-falling gait
            first_survivor = e + 1
        d_opt = float(np.linalg.norm((full_x[free] - opt[free])
                                     / (box[1] - box[0])[free]))
        rows.append(dict(episode=e + 1, mode=mode, V=V, fell=int(fell > 0.5),
                         best_V=best_V, d_opt=d_opt,
                         **{PARAM_NAMES[i]: float(full_x[i]) for i in free}))
        print(f"[ep {e+1:3d}] {mode:9s} V={V:+.3f} fell={'Y' if fell>0.5 else '.'} "
              f"| best_V={best_V:+.3f} n_safe={agent.n_safe_seen()} "
              f"| " + " ".join(f"{PARAM_NAMES[i]}={full_x[i]:.2f}" for i in free),
              flush=True)

    # summary
    best_x, best_V = agent.best()
    fall_rate_all = np.mean([r["fell"] for r in rows])
    half = len(rows) // 2
    fr_early = np.mean([r["fell"] for r in rows[:half]]) if half else float("nan")
    fr_late = np.mean([r["fell"] for r in rows[half:]]) if half else float("nan")
    print(f"\n=== GP-safe recovery summary ({a.episodes} episodes) ===")
    print(f"  incumbent V={inc_V:+.3f} (fell {inc_fell:.0%});  "
          f"oracle(reduced) V={optV:+.3f}")
    print(f"  agent best discovered V={best_V:+.3f}  "
          f"(first NON-FALLING gait at episode {first_survivor}; "
          f"{'matches/beats oracle' if best_V >= optV else 'below oracle by %.2f' % (optV-best_V)})")
    print(f"  best gait: " + " ".join(f"{PARAM_NAMES[i]}={best_x[i]:.2f}" for i in free)
          + " (frozen: " + " ".join(f"{PARAM_NAMES[i]}={best_x[i]:.2f}" for i in frozen) + ")")
    print(f"  fall rate: first half {fr_early:.0%} -> second half {fr_late:.0%} "
          f"(memory persists: {len(agent.Y)} episodes stored)")

    # csv
    import csv
    cols = ["episode", "mode", "V", "fell", "best_V", "d_opt"] + [PARAM_NAMES[i] for i in free]
    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nsaved {LOG_CSV}")
    if archive_path:
        print(f"saved memory {ARCHIVE} ({len(agent.Y)} episodes)")

    # convergence figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        ep = [r["episode"] for r in rows]
        V = [r["V"] for r in rows]
        bV = [r["best_V"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].scatter(ep, V, c=["crimson" if r["fell"] else "tab:green" for r in rows],
                      s=28, label="episode V (red=fell)")
        ax[0].plot(ep, bV, "k-", lw=1.6, label="best-so-far")
        ax[0].axhline(optV, color="tab:blue", ls="--", lw=1.2, label=f"oracle {optV:+.2f}")
        ax[0].axhline(inc_V, color="gray", ls=":", lw=1.2, label=f"incumbent {inc_V:+.2f}")
        ax[0].axhline(0, color="k", lw=0.6)
        ax[0].set_xlabel("episode"); ax[0].set_ylabel("post-damage V")
        ax[0].set_title("Discovery of a recovery gait"); ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        d = [r["d_opt"] for r in rows]
        ax[1].plot(ep, d, "o-", ms=3, color="tab:purple")
        ax[1].set_xlabel("episode"); ax[1].set_ylabel("norm. dist to oracle (reduced dims)")
        ax[1].set_title("Search approaching the stable region"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG, dpi=150, bbox_inches="tight")
        print(f"saved {FIG}")
    except Exception as ex:
        print(f"(figure skipped: {ex})")


if __name__ == "__main__":
    main()
