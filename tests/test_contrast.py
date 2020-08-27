from starry_process.contrast import ContrastIntegral
import numpy as np
from scipy.integrate import quad


def pdf(xi, mu, sigma):
    p = sigma ** 2
    q = (1 - mu) ** 2
    mu_b = np.log(q / np.sqrt(q + p))
    var_b = np.log(1 + p / q)
    b = 1 - xi
    return (
        1.0
        / (b * np.sqrt(2 * np.pi * var_b))
        * np.exp(-((np.log(b) - mu_b) ** 2) / (2 * var_b))
    )


def test_contrast(ydeg=5, mu=0.5, sigma=0.1, rtol=1e-12, ftol=1e-12):

    # Get analytic integrals
    print("Computing moments analytically...")
    I = ContrastIntegral(ydeg=ydeg)
    I._set_params(mu, sigma)
    e = I.fac1.eval()
    sqrtE = I.fac2.eval()
    E = sqrtE ** 2

    # Get the first moment by numerical integration
    print("Computing first moment numerically...")

    def func(xi):
        return xi * pdf(xi, mu, sigma)

    e_num = quad(func, -np.inf, 1)[0]

    # Get the second moment by numerical integration
    print("Computing second moment numerically...")

    def func(xi):
        return xi ** 2 * pdf(xi, mu, sigma)

    E_num = quad(func, -np.inf, 1)[0]

    # Compare
    assert np.abs(e - e_num) < rtol, "error in first moment"
    assert np.abs(1 - e / e_num) < ftol, "error in first moment"
    assert np.abs(E - E_num) < rtol, "error in second moment"
    assert np.abs(1 - E / E_num) < ftol, "error in second moment"
