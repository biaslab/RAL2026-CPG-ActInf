"""Grid-style baseline over the shared 8-D CPG parameter bounds.

Classical Cartesian grids are infeasible in 8 dimensions at the trial budget
used for the BO/MARXEFE comparison (even 2 levels per dimension would cost
256 evaluations), so this module evaluates a Latin-hypercube sample of size
`n_trials` instead. LHS gives a deterministic, space-filling design that is
the standard "grid-like" baseline in surrogate-modeling literature.

The trial pipeline (CPG rollout, objective, CSV schema) is shared with
`methods.bo_optimizer` so all three optimizers write identical CSVs that can
be compared directly.
"""

import csv
import os
import time

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

from methods.bo_optimizer import evaluate_candidate
from methods.cpg_bounds import bounds


def gridsearch_optimize_cpg(
    bounds: torch.Tensor,
    target_forward_position: float,
    robot_mass: float,
    n_trials: int = 100,
    optimizer_name: str = "GridSearch",
    seed: int = 0,
    results_dir: str = "results",
) -> tuple:
    """Latin-hypercube grid search with the shared BO/MARXEFE CSV schema."""

    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, f"{optimizer_name}_seed{seed}.csv")
    csv_columns = [
        "optimizer", "seed", "trial",
        "J", "CoT",
        "forwarddistance", "lateraldrift", "meanvx",
        "fell", "stabilityindex",
        "rmsrolldeg", "rmspitchdeg",
        "opttimesec", "simtimesec", "totaltimesec",
        "couplinggain", "wswing", "wstance",
        "FFAST", "STOPGAIN",
        "hipamplitude", "kneeamplitude", "b",
    ]
    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    lower = bounds[0].numpy()
    upper = bounds[1].numpy()
    d = bounds.shape[1]

    lhs = LatinHypercube(d=d, seed=seed)
    unit_samples = lhs.random(n=n_trials)
    samples = lower + unit_samples * (upper - lower)
    train_X = torch.tensor(samples, dtype=torch.double)

    param_names = ["couplinggain", "wswing", "wstance", "FFAST",
                   "STOPGAIN", "hipamplitude", "kneeamplitude", "b"]

    objectives        = []
    cots              = []
    forward_distances = []
    lateral_drifts    = []
    mean_velocities   = []
    stabilities       = []
    fall_flags        = []

    print("\n" + "="*70)
    print("LHS GRID SEARCH OVER CPG PARAMETERS")
    print("="*70)
    print(f"Target forward position: {target_forward_position} m (lateral target = 0)")
    print(f"Robot mass: {robot_mass} kg")
    print(f"Total trials: {n_trials}")
    print(f"Optimizer: {optimizer_name}, Seed: {seed}")
    print(f"Results: {csv_path}")
    print("="*70)

    for i in range(n_trials):
        x_np = samples[i]

        opt_start    = time.time()
        opt_time_sec = time.time() - opt_start

        J, metrics = evaluate_candidate(
            x_np, target_forward_position, robot_mass,
            optimizer_name, seed, i + 1,
        )

        metrics["opttimesec"]   = opt_time_sec
        metrics["totaltimesec"] = opt_time_sec + metrics["simtimesec"]

        csv_writer.writerow(metrics)
        csv_file.flush()

        objectives.append(J)
        cots.append(metrics["CoT"])
        forward_distances.append(metrics["forwarddistance"])
        lateral_drifts.append(metrics["lateraldrift"])
        mean_velocities.append(metrics["meanvx"])
        stabilities.append(metrics["stabilityindex"])
        fall_flags.append(metrics["fell"])

        best_J_so_far = max(objectives)
        fall_status   = "FELL" if metrics["fell"] else "OK"

        print(f"\nTrial {i+1}/{n_trials}:")
        print(f" CPG params: " + ", ".join([f"{n}={v:.3f}" for n, v in zip(param_names, x_np)]))
        print(f" Result: J = {J:.2f}, [{fall_status}], FwdDist = {metrics['forwarddistance']:.3f}m, LatDrift = {metrics['lateraldrift']:.3f}m")
        print(f" CoT = {metrics['CoT']:.3f}, Vel = {metrics['meanvx']:.3f}m/s, Stab = {metrics['stabilityindex']:.2f}°")
        print(f" Best so far: J = {best_J_so_far:.2f}")

    csv_file.close()
    print(f"\n✅ CSV results saved to: {csv_path}")

    objectives_np = np.array(objectives)
    train_Y       = torch.tensor(objectives_np, dtype=torch.double).unsqueeze(1)
    best_idx      = int(train_Y.argmax().item())
    best_params   = train_X[best_idx].numpy()
    best_J        = float(train_Y.max().item())

    forward_distances_np = np.array(forward_distances)
    fall_flags_np        = np.array(fall_flags).astype(int)
    trials               = np.arange(1, n_trials + 1)
    cumulative_falls     = np.cumsum(fall_flags_np)
    fall_rate            = cumulative_falls / trials * 100
    best_J_so_far_arr    = np.maximum.accumulate(objectives_np)

    J_thresh = 2.0
    r_thresh = 15.0
    condition = (best_J_so_far_arr >= J_thresh) & (fall_rate <= r_thresh)
    N_walk = int(trials[condition][0]) if np.any(condition) else int(n_trials)
    D_cum  = float(np.sum(forward_distances_np))

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Best objective: J = {best_J:.2f} (trial {best_idx+1})")
    print("Best parameters:")
    for name, value in zip(param_names, best_params):
        print(f" {name:16s} = {value:.4f}")
    print(f"N_walk (J >= {J_thresh}, fall_rate <= {r_thresh}%): {N_walk}")
    print(f"D_cum over {n_trials} trials: {D_cum:.3f} m")
    print("="*70)

    return train_X, train_Y, best_params, N_walk, D_cum


if __name__ == "__main__":
    train_X, train_Y, best_params, N_walk, D_cum = gridsearch_optimize_cpg(
        bounds,
        target_forward_position=4.0,
        robot_mass=10.0,
        n_trials=100,
        optimizer_name="GridSearch",
        seed=0,
    )
