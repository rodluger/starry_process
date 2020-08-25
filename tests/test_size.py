from starry_process.size import SizeIntegral
import numpy as np
from scipy.stats import beta as Beta


def test_size(ydeg=15, mu=20, sigma=5, nsamples=int(1e7), atol=2e-5):

    # Settings
    np.random.seed(0)

    # Get analytic integral
    I = SizeIntegral(ydeg=ydeg)
    I._set_params(mu, sigma)
    e = I._first_moment()
    eigE = I._second_moment()
    E = eigE @ eigE.T

    # Integrate numerically
    N = (ydeg + 1) ** 2
    y = np.zeros((N, nsamples))

    # Draw the spot size
    alpha, beta = I.transform.get_standard_params(mu, sigma)

    rho = Beta.rvs(alpha, beta, size=nsamples)

    # Compute the spot expansions
    s = I.transform.get_s(rho)

    # Empirical moments
    e_num = np.mean(s, axis=0)
    E_num = np.cov(s.T) + np.outer(e_num, e_num)

    # Compare
    assert np.allclose(e, e_num, atol=atol), "error in first moment"
    assert np.allclose(E, E_num, atol=atol), "error in second moment"
