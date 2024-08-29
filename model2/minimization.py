import os
import argparse
import pandas as pd
import logging
from likelihood import likelihood as LH
from scipy.optimize import (
    minimize,
    differential_evolution,
    dual_annealing,
    direct,
    shgo,
)

os.environ["NUMEXPR_MAX_THREADS"] = "32"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def minimization(
    bounds,
    initial_guess,
    time,
    initial_clone_count_values,
    final_clone_count_values,
    scaled_kr_values,
    method="dual_annealing",
    verbose=False,
):
    def select_likelihood_function(params):
        return LH(
            params,
            time,
            initial_clone_count_values,
            final_clone_count_values,
            scaled_kr_values,
            verbose=verbose,
        )

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
    parser = argparse.ArgumentParser(description="Optimization routine")
    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        help="Subject ID: Provide full patient id.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dual_annealing",
        help="choose from differential_evolution, dual_annealing, direct, shgo. default is dual_annealing.",
    )
    args = parser.parse_args()
    patient_id = args.patient_id
    method = args.method

    try:
        # Load patient data
        patient_data = pd.read_csv(
            f"/home/gddaslab/mxp140/TCR_project/garfinkle/new_model/{patient_id}_merged_pre_and_post_vax_for_TCR_with_fold_change_gteq1.csv",
            sep=",",
        )
    except FileNotFoundError:
        logging.error(f"File not found for patient_id: {patient_id}")
        raise

    initial_clone_count_values = patient_data["cdr3_count_x"].values
    final_clone_count_values = patient_data["cdr3_count_y"].values
    scaled_kr_values = patient_data["scaled_kr_uni_dist1"].values

    # Set time duration based on patient ID
    time_durations = {
        "subject1": 96,  # days
        "subject2": 68,  # days
        "subject3": 169,  # days
    }
    time_duration = time_durations.get(patient_id, None)
    if time_duration is None:
        raise ValueError(f"Unknown patient_id: {patient_id}")

    logging.info(f"Minimization done for: {patient_id}.")

    bounds = ((1e-08, 50),)
    initial_guess = [10]

    minimization(
        bounds=bounds,
        initial_guess=initial_guess,
        time=time_duration,
        initial_clone_count_values=initial_clone_count_values,
        final_clone_count_values=final_clone_count_values,
        scaled_kr_values=scaled_kr_values,
        method=method,
        verbose=True,
    )
