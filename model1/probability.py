import numpy as np
import warnings
from scipy.integrate import quad

# Suppress only IntegrationWarning
warnings.filterwarnings("ignore")


def integrand_lymph(tau, x1, x2, kr, M):
    p = np.exp(-x2 * kr * (x1 - tau))
    integrand_value = np.exp(-tau) * p * ((1 - p) ** (M - 1))
    return integrand_value


def integrand_tumor(tau, x1_dagger, x2_dagger, kr, M):
    p = np.exp(-x2_dagger * kr * (x1_dagger - tau))
    integrand_value = np.exp(tau) * p * ((1 - p) ** (M - 1))
    return integrand_value


def probability_lymph(x1, x2, kr, M):
    factor_up_front = 1.0 / (1 - np.exp(-x1))
    integral = quad(
        integrand_lymph,
        0,
        x1,
        args=(x1, x2, kr, M),
        limit=100,
        epsabs=1e-10,
        epsrel=1e-10,
    )[0]
    return factor_up_front * integral


def probability_tumor(x1_dagger, x2_dagger, kr, M):
    factor_up_front = 1.0 / (np.exp(x1_dagger) - 1)
    integral = quad(
        integrand_tumor,
        0,
        x1_dagger,
        args=(x1_dagger, x2_dagger, kr, M),
        limit=100,
        epsabs=1e-10,
        epsrel=1e-10,
    )[0]
    return factor_up_front * integral


def probability(x1, x2, x1_dagger, x2_dagger, omega, kr, M):
    tumor_probs = omega * probability_tumor(x1_dagger, x2_dagger, kr, M)
    lymph_probs = (1 - omega) * probability_lymph(x1, x2, kr, M)
    return tumor_probs + lymph_probs
