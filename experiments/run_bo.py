import numpy as np

from methods.bo_optimizer import bo_optimize_cpg
from methods.cpg_bounds import bounds


if __name__ == "__main__":
    RESULTS_DIR             = "results"
    TARGET_FORWARD_POSITION = 4.0   # metres along +Y to reach by end of trial
    ROBOT_MASS              = 10.0
    N_TRIALS                = 100
    N_INIT                  = 5
    SEED                    = 0

    train_X, train_Y, best, N_walk, D_cum = bo_optimize_cpg(
        bounds                  = bounds,
        target_forward_position = TARGET_FORWARD_POSITION,
        robot_mass              = ROBOT_MASS,
        n_trials                = N_TRIALS,
        n_init                  = N_INIT,
        optimizer_name          = "BO",
        seed                    = SEED,
        results_dir             = RESULTS_DIR,
    )

    J_values    = train_Y.squeeze().numpy()
    best_params = np.array(best)

    print("\n=== BO SUMMARY ===")
    print("All J values:", J_values.tolist())
    print("Best params:", best_params.tolist())
    print(f"N_walk = {N_walk}, D_cum = {D_cum:.3f} m")
