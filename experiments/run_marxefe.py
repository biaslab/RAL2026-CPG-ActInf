import os
import numpy as np

from methods.cpg_bounds import bounds
from methods.marxefe_optimizer import (
    marxefe_optimize_cpg,
    plot_marxefe_results,
)


if __name__ == "__main__":
    RESULTS_DIR             = "results"
    TARGET_FORWARD_POSITION = 4.0   # metres along +Y to reach by end of trial
    ROBOT_MASS              = 10.0
    N_TRIALS                = 100
    SEED                    = 0

    train_X, train_Y, best_params = marxefe_optimize_cpg(
        bounds,
        target_forward_position = TARGET_FORWARD_POSITION,
        robot_mass              = ROBOT_MASS,
        n_trials                = N_TRIALS,
        optimizer_name          = "MARXEFE",
        seed                    = SEED,
        results_dir             = RESULTS_DIR,
    )

    data_dir = os.path.join(RESULTS_DIR, "data")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    npz_path = os.path.join(data_dir, "marxefe_results.npz")
    np.savez(npz_path,
             train_X     = train_X.numpy(),
             train_Y     = train_Y.numpy(),
             best_params = best_params)
    print(f"\n✅ Results saved to '{npz_path}'")

    csv_path = os.path.join(RESULTS_DIR, f"MARXEFE_seed{SEED}.csv")
    plot_marxefe_results(
        csv_path                = csv_path,
        target_forward_position = TARGET_FORWARD_POSITION,
        save_prefix             = os.path.join(figures_dir, "marxefe"),
    )
    print("\n✅ All MARXEFE plots saved.")
