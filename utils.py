import numpy as np
from scipy.integrate import quad

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
    integral = quad(integrand_lymph, 0, x1, args=(x1, x2, kr, M))[0]
    return factor_up_front * integral

# def probability_lymph(x1, x2, kr, M):
#     factor_up_front = 1.0 / (1 - np.exp(-x1))
#     integral_values = np.array([quad(integrand_lymph, 0, x1_val, args=(x1_val, x2_val, kr_val, M_val), limit=80)[0] for x1_val, x2_val, kr_val, M_val in zip(x1, x2, kr, M)])
#     return factor_up_front * integral_values

def probability_tumor(x1_dagger, x2_dagger, kr, M):
    factor_up_front = 1.0 / (np.exp(x1_dagger) - 1)
    integral = quad(integrand_tumor, 0, x1_dagger, args=(x1_dagger, x2_dagger, kr, M))[0]
    return factor_up_front * integral

# def probability_tumor(x1_dagger, x2_dagger, kr, M):
#     factor_up_front = 1.0 / (np.exp(x1_dagger) - 1)
#     integral_values = np.array([quad(integrand_tumor, 0, x1_dagger_val, args=(x1_dagger_val, x2_dagger_val, kr_val, M_val), limit=80)[0] for x1_dagger_val, x2_dagger_val, kr_val, M_val in zip(x1_dagger, x2_dagger, kr, M)])
#     return factor_up_front * integral_values

def clone_dist_numerical(x1, x2, x1_dagger, x2_dagger, omega, kr, M):
    tumor_probs = omega * probability_tumor(x1_dagger, x2_dagger, kr, M)
    lymph_probs = (1 - omega) * probability_lymph(x1, x2, kr, M)
    return tumor_probs + lymph_probs
