import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from probability import log_probability

# Smallest positive float value
eps = sys.float_info.min


def likelihood(params, t, initial_dist, final_dist, scaled_kr_values, verbose=False):
    Lambda = params[0]
    n = len(scaled_kr_values)

    # Create arrays with repeated values
    Lambda_values = np.full(n, Lambda)
    t_values = np.full(n, t)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        log_probabilities = list(
            executor.map(
                log_probability,
                t_values,
                initial_dist,
                final_dist,
                Lambda_values,
                scaled_kr_values,
            )
        )
    # Convert to numpy array and flatten
    log_probabilities = np.array(log_probabilities).flatten()
    sum_log_probs = np.sum(log_probabilities)
    neg_sum = -sum_log_probs

    if verbose:
        print(f"Neg-logL:{neg_sum:.2f}     Lambda:{Lambda:.2f}")
    return neg_sum


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(
        "/home/gddaslab/mxp140/TCR_project/garfinkle/new_model/subject1_merged_pre_and_post_vax_for_TCR_with_fold_change_gteq1.csv",
        sep=",",
    )
    likelihood(
        (10,),
        123,
        df.iloc[:30, 2].values,
        df.iloc[:30, 4].values,
        df.iloc[:30, 3].values,
        verbose=True,
    )
