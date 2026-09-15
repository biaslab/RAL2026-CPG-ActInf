"""Re-score the already-run continual bouts with the CROSS-ENTROPY criterion.

Eq. (criterion) of the paper defines the locomotion objective as the cross-
entropy between the distribution the gait induces over the body outputs and the
Gaussian goal prior,

    J(theta, e) = E_p(y | theta, e) [ -ln N(y | y*, Sigma*) ]
                = 1/N sum_k [ 1/2 (y_k - y*)^T Sigma*^-1 (y_k - y*) ]
                  + 1/2 ln |2 pi Sigma*| ,                       [nats]

LOWER IS BETTER. The bouts in `results/` were scored with the older `score_V`
(saturating speed reward minus RMS tilt); this module recomputes J from the
stored per-step traces (`results/logs/<method>_seed<s>.npz`) over the SAME
evaluation windows, so no simulation has to be re-run.

Window reconstruction (identical to `continual_driver.run_event_bout`):
the window for an event opens at `detect + param_ramp*dt + 0.2` (the gait ramp-in
excluded), closes at the fall step or at the bout end, and only its trailing
`eval_hold` seconds are scored. Fall steps are read off `cum_falls`. The
reconstruction is self-checked against the `tilt_rms` recorded in
`continual_events.csv` for every event (`--check`, on by default): it reproduces
it exactly, so the windows are the ones the driver actually scored.

Two caveats, both reported by the CLI:

* `vy` IS NOT IN THE LOGS of the non-AIF arms (`continual_driver` logs
  t/y/vx/roll/pitch/...; only `continual_driver_aif` adds vy). The headline J is
  therefore the 3-D marginal over (vx, pitch, roll) for ALL arms, which is
  comparable across arms; the 4-D value including the lateral-drift term is
  reported alongside for `aif` only, as a measure of what the marginal drops.
* Falls need no sentinel here. V used V_FALL = -2.0 because a fallen bout has no
  meaningful speed reward; the cross-entropy of the real pre-fall window is
  already enormous (the trunk is tipping), so J is reported as measured, with
  survivors broken out separately.
"""

import argparse
import csv
import os

import numpy as np

# ── goal prior ───────────────────────────────────────────────────────────────
# y = (vx, vy, pitch, roll); the CONTROL/EFE goal of
# experiment-payload-adapt/run_experiment.py:271 and aif_recovery.UnifiedAIFAgent
# (NOT the loosened trigger prior, whose vx/vy std is ~1e3).
GOAL_MEAN = np.array([0.5, 0.0, 0.0, 0.0])
GOAL_STD = np.array([0.25, 0.25, np.deg2rad(12.0), np.deg2rad(12.0)])
DIMS = ("vx", "vy", "pitch", "roll")
MARGINAL = ("vx", "pitch", "roll")      # what every arm's log actually contains

# driver settings these runs used (payload experiment); see run_experiment.py
DT = 0.01
PARAM_RAMP = 30
EVAL_HOLD = 10.0
LPF_T = 0.4          # zero-phase moving average over ~one incumbent stride


def crossentropy(y, mean, std):
    """Mean over the window of -ln N(y_k | mean, diag(std^2)), in nats.
    Returns (J, per-dim mean-square terms, the log-normalizer constant)."""
    z = (np.asarray(y, float) - mean) / std
    per_dim = 0.5 * np.mean(z ** 2, axis=0)
    const = 0.5 * float(np.sum(np.log(2.0 * np.pi * std ** 2)))
    return float(np.sum(per_dim)) + const, per_dim, const


def lowpass(x, dt=DT, tau=LPF_T):
    """Zero-phase moving average over `tau` seconds (the paper's 'low-pass
    filtered angles', so the gait's own rocking is not charged as tilt)."""
    n = max(1, int(round(tau / dt)))
    if n <= 1 or len(x) < n:
        return np.asarray(x, float)
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(np.asarray(x, float), pad, mode="edge")
    return np.convolve(xp, k, mode="same")[pad:pad + len(x)]


# ── window reconstruction ────────────────────────────────────────────────────

def _fnum(v):
    return float(v) if v not in ("", None) else np.nan


def load_events(results_dir):
    with open(os.path.join(results_dir, "continual_events.csv")) as f:
        return list(csv.DictReader(f))


def event_windows(log, events, dt=DT, param_ramp=PARAM_RAMP,
                  eval_hold=EVAL_HOLD):
    """Yield (event_row, slice) per event of one (method, seed) bout, the slice
    being exactly the samples `run_event_bout` scored (empty slice -> None)."""
    t = log["t"]
    fall_k = list(np.flatnonzero(np.diff(log["cum_falls"]) > 0) + 1)
    tail_n = max(1, int(round(eval_hold / dt)))
    ramp = param_ramp * dt + 0.2
    fi = 0
    for r in events:
        detect, fell = _fnum(r["detect"]), int(r["fell"])
        if fell:
            while fi < len(fall_k) and t[fall_k[fi]] < detect:
                fi += 1                       # falls before this event's gait
            end = fall_k[fi] if fi < len(fall_k) else len(t) - 1
            fi += 1
        else:
            end = len(t) - 1
        if not np.isfinite(detect):
            yield r, None                     # fell before any gait was applied
            continue
        k0 = int(np.searchsorted(t, detect + ramp, side="right"))
        w = slice(max(k0, end + 1 - tail_n), end + 1)
        yield r, (w if w.stop > w.start else None)


def _tilt_rms_deg(roll, pitch):
    return float(np.rad2deg(np.sqrt(np.mean(roll ** 2 + pitch ** 2))))


# ── scoring ──────────────────────────────────────────────────────────────────

PRIORS = {
    # the CONTROL/EFE prior: tight vx, so a gait that survives by giving up
    # forward progress is charged for it (the paper's Eq. criterion, literally).
    "control": (("vx", "pitch", "roll"), np.array([0.5, 0.0, 0.0]),
                np.array([0.25, np.deg2rad(12.0), np.deg2rad(12.0)])),
    # the UPRIGHTNESS marginal the trigger reads (Sec. trigger): attitude only.
    "upright": (("pitch", "roll"), np.array([0.0, 0.0]),
                np.array([np.deg2rad(12.0), np.deg2rad(12.0)])),
    # loose vx: forward progress still counts, but cannot dominate attitude.
    "loose-vx": (("vx", "pitch", "roll"), np.array([0.5, 0.0, 0.0]),
                 np.array([1.0, np.deg2rad(12.0), np.deg2rad(12.0)])),
}


def score_dir(results_dir, prior="control", window="tail", lpf="angles",
              eval_hold=EVAL_HOLD, check=True):
    """Re-score every event of `results_dir`. Returns (rows, max_check_error).

    `window`: "tail" = the trailing `eval_hold` s the driver scored; "full" =
    the whole post-gait hold (onset-to-fall), which charges the tilting excursion
    but gives a surviving arm a far longer window than a falling one.
    `lpf`: which channels the moving average is applied to.
    """
    names, mean, std = PRIORS[prior]
    events = load_events(results_dir)
    bouts = {}
    for r in events:
        bouts.setdefault((r["method"], int(r["seed"])), []).append(r)

    out, worst = [], 0.0
    for (method, seed), evs in sorted(bouts.items()):
        path = os.path.join(results_dir, "logs", f"{method}_seed{seed}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as d:
            log = {k: d[k] for k in d.files}
        hold = 1e9 if window == "full" else eval_hold
        for r, w in event_windows(log, evs, eval_hold=hold):
            row = dict(method=method, seed=seed, event=int(r["event"]),
                       fell=int(r["fell"]), n_win=0, V_old=_fnum(r["V"]))
            if w is None:
                out.append(row)
                continue
            row["n_win"] = w.stop - w.start
            row["dur"] = row["n_win"] * DT
            if check and window == "tail" and np.isfinite(_fnum(r["tilt_rms"])):
                worst = max(worst, abs(_tilt_rms_deg(log["roll"][w], log["pitch"][w])
                                       - _fnum(r["tilt_rms"])))
            chans = []
            for n in names:
                x = log[n][w] if n in log else None
                if x is None:                      # vy: absent from non-AIF logs
                    chans = None
                    break
                filt = (lpf == "all") or (lpf == "angles" and n in ("pitch", "roll"))
                chans.append(lowpass(x) if filt else x)
            if chans is None:
                out.append(row)
                continue
            J, per_dim, const = crossentropy(np.column_stack(chans), mean, std)
            row.update(J=J, const=const, mean_vx=float(np.mean(log["vx"][w])))
            for n, v in zip(names, per_dim):
                row[f"J_{n}"] = float(v)
            if "vy" in log:                        # aif arm only: what vy adds
                dz = 0.5 * np.mean((lowpass(log["vy"][w]) / GOAL_STD[1]) ** 2)
                row["J_vy"] = float(dz)
            out.append(row)
    return out, worst


def summarize(rows, key="J"):
    """Per-method mean +/- SEM ACROSS SEEDS of the per-bout mean score, for all
    events and for survivors only (the seed-level aggregation
    `continual_summary.csv` uses)."""
    per_seed = {}
    for r in rows:
        if r.get("n_win") and key in r:
            per_seed.setdefault((r["method"], r["seed"]), []).append(r)
    summary = {}
    for (m, s), evs in per_seed.items():
        surv = [e[key] for e in evs if not e["fell"]]
        d = summary.setdefault(m, {"all": [], "surv": [], "n_ev": 0, "n_fall": 0})
        d["all"].append(float(np.mean([e[key] for e in evs])))
        if surv:
            d["surv"].append(float(np.mean(surv)))
        d["n_ev"] += len(evs)
        d["n_fall"] += sum(e["fell"] for e in evs)
    for m, d in summary.items():
        for k in ("all", "surv"):
            v = np.asarray(d[k], float)
            d[k + "_mean"] = float(np.mean(v)) if v.size else float("nan")
            d[k + "_sem"] = (float(np.std(v, ddof=1) / np.sqrt(v.size))
                             if v.size > 1 else float("nan"))
            d[k + "_n"] = int(v.size)
    return summary


def main():
    from methods.continual_analysis import LABELS, METHOD_ORDER
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--prior", choices=sorted(PRIORS), default="control",
                    help="which marginal of the goal prior J is taken against")
    ap.add_argument("--window", choices=("tail", "full"), default="tail",
                    help="trailing eval_hold seconds (as scored) or the whole hold")
    ap.add_argument("--lpf", choices=("angles", "all", "none"), default="angles",
                    help="channels the moving average is applied to")
    ap.add_argument("--eval-hold", type=float, default=EVAL_HOLD)
    ap.add_argument("--write", action="store_true",
                    help="write continual_crossentropy_{events,summary}.csv")
    a = ap.parse_args()

    rows, worst = score_dir(a.results_dir, prior=a.prior, window=a.window,
                            lpf=a.lpf, eval_hold=a.eval_hold)
    scored = [r for r in rows if "J" in r]
    names, mean, std = PRIORS[a.prior]
    print(f"{len(rows)} events, {len(scored)} scored | window={a.window} "
          f"(eval_hold={a.eval_hold} s) | lpf={a.lpf} ({LPF_T} s)")
    if a.window == "tail":
        print(f"window self-check vs logged tilt_rms: max |err| = {worst:.3e} deg")
    print(f"prior '{a.prior}': y* = {mean.tolist()}, std = {np.round(std, 4).tolist()} "
          f"over ({', '.join(names)});  J in nats, LOWER IS BETTER\n")

    summary = summarize(scored)
    per_dim = [f"J_{n}" for n in names]
    hdr = (f"{'method':<22}{'J (all ev)':>20}{'J (surv)':>20}"
           + "".join(f"{n:>9}" for n in per_dim) + f"{'falls/ev':>12}")
    print(hdr); print("-" * len(hdr))
    for m in METHOD_ORDER:
        if m not in summary:
            continue
        d = summary[m]
        ev = [r for r in scored if r["method"] == m]
        line = (f"{LABELS.get(m, m):<22}"
                f"{d['all_mean']:>12.2f} +/-{d['all_sem']:>5.2f}"
                f"{d['surv_mean']:>12.2f} +/-{d['surv_sem']:>5.2f}")
        line += "".join(f"{np.mean([r[c] for r in ev]):>9.2f}" for c in per_dim)
        print(line + f"{d['n_fall']:>7}/{d['n_ev']:<5}")
    print(f"\n(log-normalizer constant in J: {scored[0]['const']:+.2f} nats)")

    vy = [r for r in scored if "J_vy" in r]
    if vy and "vy" not in names:
        print(f"lateral drift (vy) is logged for the aif arm ONLY and is NOT in J; "
              f"it would add +{np.mean([r['J_vy'] for r in vy]):.2f} nats there.")
    fell = [r for r in scored if r["fell"]]
    if fell:
        print(f"falls carry NO sentinel here (V used V_FALL=-2.0): a fallen event "
              f"is scored on its real pre-fall window, mean J = "
              f"{np.mean([r['J'] for r in fell]):.2f} vs "
              f"{np.mean([r['J'] for r in scored if not r['fell']]):.2f} for survivors, "
              f"over {np.mean([r['dur'] for r in fell]):.0f} s vs "
              f"{np.mean([r['dur'] for r in scored if not r['fell']]):.0f} s of hold.")

    if a.write:
        cols = (["method", "seed", "event", "fell", "n_win", "dur", "J"]
                + per_dim + ["J_vy", "mean_vx", "const", "V_old"])
        p = os.path.join(a.results_dir, "continual_crossentropy_events.csv")
        with open(p, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader(); wr.writerows(rows)
        p2 = os.path.join(a.results_dir, "continual_crossentropy_summary.csv")
        with open(p2, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["method", "n_seeds", "J_all", "J_all_sem", "J_surv",
                         "J_surv_sem", "n_events", "n_falls", "prior", "window"])
            for m in METHOD_ORDER:
                if m in summary:
                    d = summary[m]
                    wr.writerow([m, d["all_n"], d["all_mean"], d["all_sem"],
                                 d["surv_mean"], d["surv_sem"], d["n_ev"],
                                 d["n_fall"], a.prior, a.window])
        print(f"\nwrote {p}\n      {p2}")


if __name__ == "__main__":
    main()
