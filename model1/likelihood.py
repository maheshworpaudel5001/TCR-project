from probability import probability
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os


def likelihood(params, clone_count_values, scaled_kr_values, verbose=False):
    x1, x2, x2_dagger, omega = params  # We set x1==x1_dagger.
    M_values = clone_count_values
    n = len(clone_count_values)

    # Create arrays once
    x1_values = np.full(n, x1)
    x2_values = np.full(n, x2)
    x2_dagger_values = np.full(n, x2_dagger)
    omega_values = np.full(n, omega)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        probabilities = list(
            executor.map(
                probability,
                x1_values,
                x2_values,
                x1_values,
                x2_dagger_values,
                omega_values,
                scaled_kr_values,
                M_values,
            )
        )

    sum_log_probs = np.sum(np.log(probabilities))
    neg_sum = -sum_log_probs

    if verbose:
        print(
            f"Neg-logL:{neg_sum:.2f}     x1:{x1:.2f}, x2:{x2:.2f}, x2_dagger:{x2_dagger:.2f}, omega:{omega:.2f}"
        )
    return neg_sum


if __name__ == "__main__":
    # Sample parameters
    params = (0.5, 1.0, 1.5, 0.1)

    # Sample data
    clone_count_values = np.array([10, 20, 30, 40, 50] * 100)
    scaled_kr_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 100)

    # Run the likelihood function
    result = likelihood(params, clone_count_values, scaled_kr_values, verbose=True)
