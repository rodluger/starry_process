from starry_gp.size import get_s, SizeIntegral
from starry_gp.transforms import get_alpha_beta
import numpy as np
from scipy.stats import beta as Beta


def test_size(ydeg=5, mu=0.1, nu=0.1):

    # Settings
    np.random.seed(0)
    nsamples = int(1e7)
    atol = 5.0e-6

    # Get analytic integral
    I = SizeIntegral(ydeg=ydeg)
    I._set_params(mu, nu)
    e = I._first_moment()
    eigE = I._second_moment()
    E = eigE @ eigE.T

    # Integrate numerically
    N = (ydeg + 1) ** 2
    y = np.zeros((N, nsamples))

    # Draw the spot size
    alpha, beta = get_alpha_beta(mu, nu)
    r = Beta.rvs(alpha, beta, size=nsamples)

    # Compute the spot expansions
    s = get_s(ydeg, r)

    # Empirical moments
    e_num = np.mean(s, axis=0)
    E_num = np.cov(s.T) + np.outer(e_num, e_num)

    # Compare
    assert np.allclose(e, e_num, atol=atol), "error in first moment"
    assert np.allclose(E, E_num, atol=atol), "error in second moment"
