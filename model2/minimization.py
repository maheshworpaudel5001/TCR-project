import os

os.environ["NUMEXPR_MAX_THREADS"] = "32"

import logging
from model2.likelihood import likelihood as LH
from scipy.optimize import (
    minimize,
    differential_evolution,
    dual_annealing,
    direct,
    shgo,
)
from pyswarm import pso


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
    elif method == "PSO":
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        result = pso(select_likelihood_function, lb, ub, swarmsize=100, maxiter=200)
        result = {"x": result[0], "success": True, "message": "PSO completed"}
    else:
        raise ValueError(f"Unknown method: {method}")

    if result["success"]:
        logging.info("Minimization successful")
    else:
        logging.warning("Minimization failed")

    optimum_values = result["x"]
    logging.info(result["message"])

    return optimum_values, select_likelihood_function(optimum_values)
