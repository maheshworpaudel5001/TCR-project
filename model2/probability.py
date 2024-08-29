from scipy.special import comb as comb
import numpy as np
from scipy.special import loggamma


def probability(t, n0, n, Lambda, kr):
    """Probability of a TCR with propensity kr to interact with neo-antigens to have frequency n at time t from n0 at time t=0

    Args:
        t (float): time at which probability is calculated
        n0 (int/float): initial frequency of the TCR at t=0
        n (int/float): final frequency of the TCR at t=t
        Lambda (float): scaling constant determined as parameter from optimization
        kr (float): provided from data

    Returns:
        float: probability
    """
    f = comb(n - 1, n0 - 1)
    exp1 = np.exp(-n0 * Lambda * kr * t)
    exp2 = 1 - np.exp(-Lambda * kr * t)
    prob = f * exp1 * (exp2 ** (n - n0))
    return prob


def log_probability(t, n0, n, Lambda, kr):
    """
    log_probability: log of the probability calculated from above function

    t (float): time at which probability is calculated
        n0 (int/float): initial frequency of the TCR at t=0
        n (int/float): final frequency of the TCR at t=t
        Lambda (float): scaling constant determined as parameter from optimization
        kr (float): provided from data

    Returns:
        float: log of probability
    """
    log_f1 = loggamma(n)
    log_f2 = loggamma(n - n0 + 1)
    log_f3 = loggamma(n0)
    log_f = (
        log_f1 - log_f2 - log_f3
    )  # Log of combinatorics factor with Stirling's approximation applied
    log_exp1 = -n0 * Lambda * kr * t
    log_exp2 = (n - n0) * np.log(1 - np.exp(-Lambda * kr * t))
    log_prob = log_f + log_exp1 + log_exp2
    return log_prob
