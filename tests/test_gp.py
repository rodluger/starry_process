from starry_gp.gp import YlmGP
from starry_gp.transform import get_alpha_beta
from numerical import spot
import numpy as np
from tqdm import tqdm
from scipy.stats import beta as Beta
from scipy.stats import lognorm as LogNormal


def test_moments():

    # Settings
    ydeg = 5
    atol = 5e-2
    mu_lat = 0.9
    nu_lat = 0.1
    mu_r = 0.05
    nu_r = 0.1
    mu_b = np.log(0.3)
    nu_b = 0.1
    np.random.seed(0)
    nsamples = 100000

    # Integrate analytically
    gp = YlmGP(ydeg)
    gp.set_params(mu_lat, nu_lat, mu_r, nu_r, mu_b, nu_b)
    mu = gp.mu
    cov = gp.cov

    # Integrate numerically
    y = np.zeros((nsamples, (ydeg + 1) ** 2))
    alpha_lat, beta_lat = get_alpha_beta(mu_lat, nu_lat)
    alpha_r, beta_r = get_alpha_beta(mu_r, nu_r)
    for n in tqdm(range(nsamples)):

        # Draw the spot size
        r = Beta.rvs(alpha_r, beta_r, size=nsamples)

        # Draw the spot amplitude
        delta = 1 - LogNormal.rvs(s=np.sqrt(nu_b), scale=np.exp(mu_b), size=nsamples)

        # Draw the latitude
        lat = np.arccos(Beta.rvs(alpha_lat, beta_lat)) * 180.0 / np.pi
        lat *= 2.0 * (int(np.random.random() > 0.5) - 0.5)

        # Draw the longitude
        lon = 360.0 * np.random.random()

        # Compute the spot expansion (TODO)
        raise NotImplementedError("TODO!")

    mu_num = np.mean(y, axis=0)[1:]
    cov_num = np.cov(y.T)[1:, 1:]

    # Compare
    assert np.allclose(mu, mu_num, atol=atol)
    assert np.allclose(cov, cov_num, atol=atol)


if __name__ == "__main__":
    test_moments()
