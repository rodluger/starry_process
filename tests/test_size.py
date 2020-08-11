from starry_gp.size import SizeIntegral
from starry_gp.transform import get_alpha_beta
from numerical import spot
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta as Beta
from scipy.stats import lognorm
from tqdm import tqdm


def test_first_moment(ydeg=5, mu_r=0.1, nu_r=0.01, mu_b=-0.25, nu_b=1.0):

    np.random.seed(0)
    nsamples = 1000000
    atol = 2e-4

    # Get analytic integral
    S = SpotIntegral(ydeg)
    S.set_params(mu_r, nu_r, mu_b, nu_b)
    mu = S.first_moment()

    # Integrate numerically
    N = (ydeg + 1) ** 2
    y = np.zeros((N, nsamples))

    # Draw the spot size
    alpha_r, beta_r = get_alpha_beta(mu_r, nu_r)
    r = Beta.rvs(alpha_r, beta_r, size=nsamples)

    # Draw the spot amplitude
    delta = 1 - lognorm.rvs(s=np.sqrt(nu_b), scale=np.exp(mu_b), size=nsamples)

    # Compute the spot expansions
    s = spot(ydeg, r, delta, c=1.0)
    mu_num = np.mean(s, axis=0)

    # Compare
    assert np.allclose(mu, mu_num, atol=atol)


def test_second_moment(ydeg=5, mu_r=0.1, nu_r=0.01, mu_b=-0.25, nu_b=1.0):

    np.random.seed(0)
    nsamples = 1000000
    atol = 2e-4

    # Compute the analytic covariance
    S = SpotIntegral(ydeg)
    S.set_params(mu_r, nu_r, mu_b, nu_b)
    mu = S.first_moment().reshape(-1, 1)
    C = S.second_moment()
    K = C @ C.T - mu @ mu.T

    # Integrate numerically
    N = (ydeg + 1) ** 2
    y = np.zeros((N, nsamples))

    # Draw the spot size
    alpha_r, beta_r = get_alpha_beta(mu_r, nu_r)
    r = Beta.rvs(alpha_r, beta_r, size=nsamples)

    # Draw the spot amplitude
    delta = 1 - lognorm.rvs(s=np.sqrt(nu_b), scale=np.exp(mu_b), size=nsamples)

    # Compute the spot expansions
    s = spot(ydeg, r, delta, c=1.0)
    K_num = np.cov(s.T)

    # Compare
    assert np.allclose(K, K_num, atol=atol)


if __name__ == "__main__":
    test_first_moment()
    test_second_moment()
