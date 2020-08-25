from starry_process.gp import YlmGP
from starry_process.transforms import get_alpha_beta
from starry_process.wigner import R
from starry_process.size import get_s
import numpy as np
from tqdm import tqdm
from scipy.stats import beta as Beta
from scipy.stats import lognorm as LogNormal


def test_moments():

    # Settings
    ydeg = 5
    atol = 1e-4
    size = (0.1, 0.1)  # mean, variance
    contrast = (-0.1, 0.01)  # mean, variance
    latitude = (0.5, 0.1)  # mean, variance
    np.random.seed(0)
    nsamples = int(1e5)

    # Integrate analytically
    gp = YlmGP(ydeg)
    gp.size.set_params(*size)
    gp.contrast.set_params(*contrast)
    gp.latitude.set_params(*latitude)
    mu = gp.mean
    cov = gp.cov

    # Integrate numerically

    # Draw the spot size
    alpha_r, beta_r = get_alpha_beta(*size)
    r = Beta.rvs(alpha_r, beta_r, size=nsamples)

    # Draw the spot amplitude
    xi = 1 - LogNormal.rvs(
        scale=np.exp(contrast[0]), s=np.sqrt(contrast[1]), size=nsamples
    )

    # Draw the latitude
    alpha_lat, beta_lat = get_alpha_beta(*latitude)
    lat = np.arccos(Beta.rvs(alpha_lat, beta_lat, size=nsamples))
    lat *= 2.0 * (
        np.array(np.random.random(size=nsamples) > 0.5, dtype=int) - 0.5
    )

    # Draw the longitude
    lon = 2 * np.pi * np.random.random(size=nsamples)

    # Integrate numerically by sampling
    N = (ydeg + 1) ** 2
    y = np.empty((nsamples, N))
    for n in tqdm(range(nsamples)):

        # Compute the spot expansion at (0, 0)
        s = get_s(ydeg, r[n]).reshape(-1)

        # Rotation in latitude
        Rx = R(
            ydeg,
            phi=lat[n],
            cos_alpha=0,
            sin_alpha=1,
            cos_gamma=0,
            sin_gamma=-1,
        )

        # Rotation in longitude
        Ry = R(
            ydeg,
            phi=lon[n],
            cos_alpha=1,
            sin_alpha=0,
            cos_gamma=1,
            sin_gamma=0,
        )

        # Apply the transformations
        for l in range(ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            y[n, i] = xi[n] * (Ry[l] @ (Rx[l] @ s[i]))

    mu_num = np.mean(y, axis=0)
    cov_num = np.cov(y.T)

    # Compare
    assert np.allclose(mu, mu_num, atol=atol)
    assert np.allclose(cov, cov_num, atol=atol)
