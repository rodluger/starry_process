from starry_gp.spot import SpotIntegral
from numerical import spot
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm as Normal
from tqdm import tqdm


def test_first_moment(
    ydeg=3, mu_lns=-3.0, sig_lns=0.1, mu_lna=-2.3, sig_lna=0.1, sign=-1
):

    np.random.seed(0)
    nsamples = 100000
    atol = 1e-4

    # Get analytic integral
    S = SpotIntegral(ydeg)
    S.set_params(
        mu_lns=mu_lns, sig_lns=sig_lns, mu_lna=mu_lna, sig_lna=sig_lna, sign=sign
    )
    mu = S.first_moment()

    # Integrate numerically
    N = (ydeg + 1) ** 2
    mu_num = np.zeros(N)
    lnsigma = mu_lns + sig_lns * np.random.randn(nsamples)
    sigma = np.exp(lnsigma)
    lnamp = mu_lna + sig_lna * np.random.randn(nsamples)
    amp = 1 / (1 + np.exp(-lnamp))
    s = np.empty((nsamples, N))
    for k in tqdm(range(nsamples)):
        s[k] = spot(ydeg, sigma[k], amp[k], sign)
    mu_num = np.mean(s, axis=0).reshape(-1)

    # Compare
    assert np.allclose(mu, mu_num, atol=atol)


def test_second_moment(
    ydeg=3, mu_lns=-3.0, sig_lns=0.1, mu_lna=-2.3, sig_lna=0.1, sign=-1
):

    np.random.seed(0)
    nsamples = 100000
    atol = 1e-4

    # Compute the analytic covariance
    S = SpotIntegral(ydeg)
    S.set_params(
        mu_lns=mu_lns, sig_lns=sig_lns, mu_lna=mu_lna, sig_lna=sig_lna, sign=sign
    )
    mu = S.first_moment().reshape(-1, 1)
    C = S.second_moment()
    K = C @ C.T - mu @ mu.T

    # Integrate numerically
    N = (ydeg + 1) ** 2
    lnsigma = mu_lns + sig_lns * np.random.randn(nsamples)
    sigma = np.exp(lnsigma)
    lnamp = mu_lna + sig_lna * np.random.randn(nsamples)
    amp = 1 / (1 + np.exp(-lnamp))
    s = np.empty((nsamples, N))
    for k in tqdm(range(nsamples)):
        s[k] = spot(ydeg, sigma[k], amp[k], sign)
    mu_num = np.mean(s, axis=0).reshape(-1, 1)
    K_num = np.cov(s.T)

    # Compare
    assert np.allclose(K, K_num, atol=atol)


if __name__ == "__main__":
    test_first_moment()
    test_second_moment()
