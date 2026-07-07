"""Transition-recovery analysis for the friction experiment.

For each seed we replayed each method's best gait and saved (t, vx, y) traces
plus the friction-zone boundaries. Here we measure, for every friction boundary
the robot actually crosses, how well it holds the target forward velocity just
after the transition:

  * post-transition velocity error = mean |vx - v*| over a 0.8 s window after
    the crossing (lower = recovers/holds speed better);
  * recovery time = time after the crossing for |vx - v*| to fall below 0.2 m/s
    and stay there for 0.15 s (censored at the 0.8 s window if it never does).

Per-episode methods (BO, grid) keep fixed parameters and can only recover via
passive dynamics; the per-cycle MARX-EFE agent re-tunes online, so it should
hold speed better across friction changes.

Usage (from repo root):  python experiment-friction/transition_recovery.py
"""

import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(_HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
MIN_DROP = 0.15      # only count transitions where friction drops by at least this
METHODS = [("GridSearch", "tab:green"), ("BO", "tab:blue"), ("MARXEFE", "tab:orange")]

WINDOW_S = 0.8       # analysis window after each transition
REC_BAND = 0.2       # |vx - v*| band counted as "recovered" (m/s)
REC_HOLD_S = 0.15    # must stay within band this long to count as recovered
SETTLE_S = 1.0       # ignore transitions during the initial settling window


def _crossing_indices(t, y, zones_y, zones_mu, base_mu):
    """Indices where the robot first crosses a friction-DROP boundary (μ
    decreases by >= MIN_DROP), with enough trace on both sides for the pre/post
    windows. Friction-increase and tiny-change boundaries are ignored — only
    drops onto a slipperier surface are real disturbances to recover from."""
    dt = t[1] - t[0]
    win = int(WINDOW_S / dt)
    out = []
    for i, yb in enumerate(zones_y):
        mu_before = base_mu if i == 0 else zones_mu[i - 1]
        mu_after = zones_mu[i]
        if (mu_before - mu_after) < MIN_DROP:
            continue                       # not a (sufficient) friction drop
        if yb <= y[0] or yb >= y[-1]:
            continue                       # boundary not traversed
        k = int(np.argmax(y >= yb))
        if t[k] < SETTLE_S or k + win >= len(t) or k - win < 0:
            continue
        out.append((yb, k))
    return out, win


def analyze():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "traces_seed*.npz")),
                   key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
    if not files:
        raise SystemExit(f"No traces in {RESULTS_DIR}. Run run_multiseed first.")

    per_method = {m: {"err": [], "rec": [], "dlt": [], "n": 0} for m, _ in METHODS}

    for f in files:
        d = np.load(f)
        v_star = float(d["target_v"])
        zones_y = d["zones_y"]
        zones_mu = d["zones_mu"]
        base_mu = float(d["base_mu"])
        for method, _ in METHODS:
            if f"{method}_t" not in d:
                continue
            t, vx, y = d[f"{method}_t"], d[f"{method}_vx"], d[f"{method}_y"]
            if len(t) < 5:
                continue
            dt = t[1] - t[0]
            crossings, win = _crossing_indices(t, y, zones_y, zones_mu, base_mu)
            hold = int(REC_HOLD_S / dt)
            for _yb, k in crossings:
                seg = vx[k:k + win]
                post_err = np.mean(np.abs(seg - v_star))
                pre_err = np.mean(np.abs(vx[k - win:k] - v_star))
                per_method[method]["err"].append(post_err)
                # transition-INDUCED error: post minus pre window (removes the
                # gait's baseline speed offset and stride oscillation).
                per_method[method]["dlt"].append(post_err - pre_err)
                within = np.abs(seg - v_star) < REC_BAND
                rec = WINDOW_S
                for j in range(len(within) - hold):
                    if within[j:j + hold].all():
                        rec = j * dt
                        break
                per_method[method]["rec"].append(rec)
                per_method[method]["n"] += 1

    print("\n=== transition-recovery (mean ± std over all crossed boundaries) ===")
    print(f"{'method':11s} {'#trans':>7} {'postErr':>9} {'d_err(post-pre)':>16} "
          f"{'recTime[s]':>11}")
    summary = {}
    for method, _ in METHODS:
        e = np.array(per_method[method]["err"])
        dl = np.array(per_method[method]["dlt"])
        r = np.array(per_method[method]["rec"])
        if e.size == 0:
            print(f"{method:11s} {'0':>7}")
            continue
        summary[method] = (e.mean(), e.std(), r.mean(), r.std(), e.size,
                           dl.mean(), dl.std())
        print(f"{method:11s} {e.size:>7d} {e.mean():>5.3f}±{e.std():.3f} "
              f"{dl.mean():>+7.3f}±{dl.std():.3f} {r.mean():>6.3f}±{r.std():.3f}")

    # ---- Figure 1: bar chart of post-transition velocity error ----
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (axb, axt) = plt.subplots(1, 2, figsize=(12, 5))
    names = [m for m, _ in METHODS if m in summary]
    colors = [c for m, c in METHODS if m in summary]
    dlts = [summary[m][5] for m in names]
    dltsd = [summary[m][6] for m in names]
    recs = [summary[m][2] for m in names]
    recsd = [summary[m][3] for m in names]
    axb.bar(names, dlts, yerr=dltsd, color=colors, alpha=0.85, capsize=5)
    axb.axhline(0, color="k", lw=0.8)
    axb.set_ylabel("Transition-induced velocity error  Δ(post − pre) [m/s]")
    axb.set_title("Extra velocity error caused by the transition (lower = better)")
    axb.grid(True, axis="y", alpha=0.3)
    axt.bar(names, recs, yerr=recsd, color=colors, alpha=0.85, capsize=5)
    axt.set_ylabel("Recovery time [s]")
    axt.set_title(f"Time to recover within {REC_BAND} m/s (capped at {WINDOW_S}s)")
    axt.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Transition recovery across random friction zones "
                 f"({len(files)} seeds)", fontweight="bold")
    fig.tight_layout()
    out1 = os.path.join(FIG_DIR, "transition_recovery.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out1}")

    # ---- Figure 2: representative velocity trace with transitions marked ----
    # Pick the seed whose terrain has the most boundaries within reach.
    best_f, best_n = None, -1
    for f in files:
        d = np.load(f)
        y = d.get("MARXEFE_y")
        if y is None:
            continue
        n = sum(1 for yb in d["zones_y"] if y[0] < yb < y[-1])
        if n > best_n:
            best_f, best_n = f, n
    if best_f is not None:
        d = np.load(best_f)
        sd = re.search(r"seed(\d+)", best_f).group(1)
        plt.figure(figsize=(10, 5))
        for method, color in METHODS:
            if f"{method}_t" in d:
                plt.plot(d[f"{method}_t"], d[f"{method}_vx"], color=color,
                         label=method, lw=1.5)
        plt.axhline(float(d["target_v"]), color="k", ls=":", lw=1, label="target v*")
        for yb in d["zones_y"]:
            ym = d.get("MARXEFE_y")
            if ym is not None and ym[0] < yb < ym[-1]:
                k = int(np.argmax(ym >= yb))
                plt.axvline(d["MARXEFE_t"][k], color="gray", ls="--", lw=1, alpha=0.6)
        plt.xlabel("Time [s]"); plt.ylabel("Forward velocity vx [m/s]")
        plt.title(f"Forward velocity vs time (seed {sd}); dashed = friction "
                  f"transitions (MARX-EFE crossings)")
        plt.legend(loc="best"); plt.grid(True, alpha=0.3)
        out2 = os.path.join(FIG_DIR, "transition_recovery_trace.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"saved {out2}")


if __name__ == "__main__":
    analyze()
