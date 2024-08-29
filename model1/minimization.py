import os

os.environ["NUMEXPR_MAX_THREADS"] = "32"

import logging
from likelihood import likelihood as LH
from scipy.optimize import (
    minimize,
    differential_evolution,
    dual_annealing,
    direct,
    shgo,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def minimization(
    bounds,
    initial_guess,
    clone_count_values,
    scaled_kr_values,
    method="dual_annealing",
    verbose=False,
):
    def select_likelihood_function(params):
        return LH(params, clone_count_values, scaled_kr_values, verbose=verbose)

    logging.info(f"These are bounds: {bounds}")

    # Ensure that the initial guess falls within the bounds
    for i, (lower_bound, upper_bound) in enumerate(bounds):
        if not (lower_bound <= initial_guess[i] <= upper_bound):
            raise ValueError(
                f"Initial guess for parameter {i} is not within the bounds."
            )

    logging.info("Minimization started")
    logging.info(f"Method: {method}")

    if method == "dual_annealing":
        result = dual_annealing(
            select_likelihood_function, bounds, x0=initial_guess, seed=1
        )
    elif method == "direct":
        result = direct(select_likelihood_function, bounds)
    elif method == "differential_evolution":
        result = differential_evolution(
            select_likelihood_function, bounds, x0=initial_guess, disp=True, workers=1
        )
    elif method == "shgo":
        result = shgo(select_likelihood_function, bounds)
    elif method == "SLSQP":
        result = minimize(
            select_likelihood_function, initial_guess, bounds=bounds, method=method
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if result.success:
        logging.info("Minimization successful")
    else:
        logging.warning("Minimization failed")

    optimum_values = result.x
    logging.info(result.message)

    return optimum_values, select_likelihood_function(optimum_values)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Optimization routine")
    parser.add_argument("--class_id", type=str, help="Class ID: either brmet or gbm.")
    parser.add_argument(
        "--patient_id",
        type=str,
        help="Patient ID or if all of patients write all_(class_id)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dual_annealing",
        help="choose from differential_evolution, dual_annealing, direct, shgo. default is dual_annealing.",
    )
    args = parser.parse_args()
    class_id = args.class_id
    patient_id = args.patient_id
    method = args.method

    full_data = pd.read_csv(
        f"/home/gddaslab/mxp140/TCR_project/data/full_{class_id}_data_v2.csv",
        sep=",",
        comment="#",
    )
    if "all" not in patient_id:
        patient_data = full_data[
            full_data["CDR3"].str.contains(patient_id)
        ].reset_index()
    else:
        patient_data = full_data
    clone_count_values = patient_data["cdr3_count"].values
    scaled_kr_values = patient_data["scaled_kr"].values

    logging.info(f"Minimization done for: {patient_id}.")
    bounds = (
        (1e-08, 500),
        (1e-08, 500),
        (1e-08, 500),
        (1e-08, 0.5),
    )

    initial_guess_map = {
        "brmet008": [5, 5, 5, 0.01],
        "brmet018": [5, 5, 5, 0.01],
        "brmet019": [5, 5, 5, 0.01],
        "brmet025": [5, 5, 5, 0.01],
        "brmet028": [5, 5, 5, 0.1],
        "gbm059": [5, 5, 5, 0.1],
        "gbm064": [5, 5, 5, 0.1],
        "gbm079": [5, 5, 5, 0.1],
        "brmet009": [5, 5, 5, 0.02],
        "brmet027": [5, 5, 5, 0.3],
    }
    initial_guess = initial_guess_map.get(patient_id, [5, 5, 5, 0.4])

    minimization(
        bounds=bounds,
        initial_guess=initial_guess,
        clone_count_values=clone_count_values,
        scaled_kr_values=scaled_kr_values,
        method=method,
        verbose=True,
    )
