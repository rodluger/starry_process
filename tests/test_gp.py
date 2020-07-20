from starry_gp.gp import YlmGP
from numerical import get_spot_function
import numpy as np
from tqdm import tqdm
from scipy.stats import beta as Beta


def test_moments():

    # Settings
    ydeg = 5
    atol = 5e-2
    mu_beta = 0.5
    nu_beta = 0.01
    mu_lns = np.log(0.1)
    sig_lns = 0.1
    mu_lna = np.log(0.2)
    sig_lna = 0.1
    np.random.seed(0)
    nsamples = 100000

    # Integrate analytically
    gp = YlmGP(ydeg)
    gp.set_params(mu_beta, nu_beta, mu_lns, sig_lns, mu_lna, sig_lna)
    mu = gp.mu
    cov = gp.cov

    # Integrate numerically
    spot = get_spot_function(ydeg)
    y = np.zeros((nsamples, (ydeg + 1) ** 2))
    alpha = mu_beta * (1 / nu_beta - 1)
    beta = (1 - mu_beta) * (1 / nu_beta - 1)
    for n in tqdm(range(nsamples)):
        lnsigma = mu_lns + sig_lns * np.random.randn()
        sigma = np.exp(lnsigma)
        lnamp = mu_lna + sig_lna * np.random.randn()
        amp = 1 / (1 + np.exp(-lnamp))
        lat = np.arccos(Beta.rvs(alpha, beta)) * 180.0 / np.pi
        lat *= 2.0 * (int(np.random.random() > 0.5) - 0.5)
        lon = 360.0 * np.random.random()
        y[n] = spot(amp, sigma, lat, lon)
    mu_num = np.mean(y, axis=0)[1:]
    cov_num = np.cov(y.T)[1:, 1:]

    # Compare
    assert np.allclose(mu, mu_num, atol=atol)
    assert np.allclose(cov, cov_num, atol=atol)


if __name__ == "__main__":
    test_moments()
