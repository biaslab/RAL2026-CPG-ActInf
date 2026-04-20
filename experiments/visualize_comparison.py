import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

RESULTS_DIR = "results"
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
SAVE_FIG    = True
SEED        = 0

METHODS = [
    ("BO",         f"BO_seed{SEED}.csv"),
    ("MARXEFE",    f"MARXEFE_seed{SEED}.csv"),
    ("GridSearch", f"GridSearch_seed{SEED}.csv"),
]

REQUIRED_COLS = [
    "trial", "J", "forwarddistance", "lateraldrift",
    "stabilityindex", "fell", "totaltimesec",
]

# (column, y-axis label) — plotted both vs trial and vs cumulative wall-clock time
METRICS = [
    ("J",               "Objective J"),
    ("forwarddistance", "Forward distance Y [m]"),
    ("lateraldrift",    "Lateral drift X [m]"),
    ("stabilityindex",  "Stability index [deg]"),
]


# ---------------------------------------------------------------------
# LOADING + SUMMARY
# ---------------------------------------------------------------------

def load_runs():
    runs = {}
    for label, fname in METHODS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"{label} CSV missing columns: {missing}")
        df = df.sort_values("trial").reset_index(drop=True)
        df["cumulative_time_sec"] = df["totaltimesec"].cumsum()
        runs[label] = df
    return runs


def summarize_method(name, df, D_star=1.0, S_star=20.0, drift_star=0.3):
    print(f"\n===== {name} SUMMARY =====")
    best = df.loc[df["J"].idxmax()]
    print(f"Trials: {len(df)}")
    print(f"Best J: {best['J']:.3f} at trial {int(best['trial'])}")
    print(f"Best forward distance (Y): {best['forwarddistance']:.3f} m")
    print(f"Best lateral drift (X): {best['lateraldrift']:.3f} m")
    print(f"Best stability_index: {best['stabilityindex']:.2f} deg")
    print(f"Total falls: {df['fell'].sum()} ({100*df['fell'].mean():.1f} %)")

    good = (
        (df["forwarddistance"]     > D_star)     &
        (df["stabilityindex"]      < S_star)     &
        (df["fell"] == 0)                        &
        (df["lateraldrift"].abs()  < drift_star)
    )
    if good.any():
        idx = good.idxmax()
        print(f"First good gait (fwd>{D_star}m, |drift|<{drift_star}m, S<{S_star}, no fall):")
        print(f"  trial {int(df.loc[idx, 'trial'])}, "
              f"cumulative_time = {df.loc[idx, 'cumulative_time_sec']:.1f} s")
    else:
        print(f"No trial reached threshold (fwd>{D_star}m, |drift|<{drift_star}m, S<{S_star}, no fall).")


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def save_fig(fname):
    if not SAVE_FIG:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric(runs, x_col, x_label, y_col, y_label, title, fname):
    plt.figure(figsize=(10, 6))
    for label, df in runs.items():
        plt.plot(df[x_col], df[y_col], "o-", label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(fname)


def plot_fall_rate(runs):
    plt.figure(figsize=(10, 6))
    for label, df in runs.items():
        trials = df["trial"].values
        falls  = df["fell"].values.astype(int)
        plt.plot(trials, np.cumsum(falls) / trials * 100.0, "-", label=label)
    plt.xlabel("Trial")
    plt.ylabel("Cumulative fall rate [%]")
    plt.title("Fall rate vs Trial")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig("fall_rate_vs_trial.png")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    t0 = time.time()
    runs = load_runs()
    if not runs:
        raise RuntimeError(f"No CSVs found in {RESULTS_DIR}")

    for label, df in runs.items():
        summarize_method(label, df)

    for y_col, y_label in METRICS:
        plot_metric(runs, "trial", "Trial",
                    y_col, y_label,
                    f"{y_label} vs Trial",
                    f"{y_col}_vs_trial.png")
        plot_metric(runs, "cumulative_time_sec", "Cumulative wall-clock time [s]",
                    y_col, y_label,
                    f"{y_label} vs Wall-clock Time",
                    f"{y_col}_vs_time.png")
    plot_fall_rate(runs)

    if SAVE_FIG:
        print(f"\nFigures saved to: {FIG_DIR}")
    else:
        plt.show()
    print(f"Analysis finished in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
