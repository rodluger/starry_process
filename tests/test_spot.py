from starry_gp.spot import SpotIntegral
from starry_gp.transform import get_alpha_beta
from numerical import spot
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta as Beta
from scipy.stats import lognorm
from tqdm import tqdm


def test_first_moment(ydeg=5, mu_s=0.1, nu_s=0.01, mu_a=-3.0, nu_a=1.0):

    np.random.seed(0)
    nsamples = 10000000
    atol = 1e-4

    # Get analytic integral
    S = SpotIntegral(ydeg)
    S.set_params(mu_s, nu_s, mu_a, nu_a)
    mu = S.first_moment()

    # Integrate numerically
    N = (ydeg + 1) ** 2
    y = np.zeros((N, nsamples))

    # Draw the spot size
    alpha_s, beta_s = get_alpha_beta(mu_s, nu_s)
    s = Beta.rvs(alpha_s, beta_s, size=nsamples)

    # Draw the spot amplitude
    a = lognorm.rvs(s=np.sqrt(nu_a), scale=np.exp(mu_a), size=nsamples)

    # Compute the spot expansions
    for l in tqdm(range(1, ydeg + 1)):
        y[l * (l + 1)] = -a * (1 + s) ** (-(l ** 2))
    mu_num = np.mean(y, axis=1).reshape(-1)

    # Compare
    assert np.allclose(mu, mu_num, atol=atol)


def test_second_moment(ydeg=5, mu_s=0.1, nu_s=0.01, mu_a=-3.0, nu_a=1.0):

    np.random.seed(0)
    nsamples = 10000000
    atol = 1e-4

    # Compute the analytic covariance
    S = SpotIntegral(ydeg)
    S.set_params(mu_s, nu_s, mu_a, nu_a)
    mu = S.first_moment().reshape(-1, 1)
    C = S.second_moment()
    K = C @ C.T - mu @ mu.T

    # Integrate numerically
    N = (ydeg + 1) ** 2
    y = np.zeros((N, nsamples))

    # Draw the spot size
    alpha_s, beta_s = get_alpha_beta(mu_s, nu_s)
    s = Beta.rvs(alpha_s, beta_s, size=nsamples)

    # Draw the spot amplitude
    a = lognorm.rvs(s=np.sqrt(nu_a), scale=np.exp(mu_a), size=nsamples)

    # Compute the spot expansions
    for l in tqdm(range(1, ydeg + 1)):
        y[l * (l + 1)] = -a * (1 + s) ** (-(l ** 2))
    K_num = np.cov(y)

    # Compare
    assert np.allclose(K, K_num, atol=atol)


if __name__ == "__main__":
    test_first_moment()
    test_second_moment()
