from starry_process import StarryProcess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytest

starry = pytest.importorskip("starry")


def get_numerical_mean_and_cov(t, nsamples=10000):
    """
    Compute the empirical covariance of the flux
    marginalized over inclination.

    """
    # Draw random Ylm samples
    sp = StarryProcess()
    mu = sp.mean_ylm.eval()
    cho_cov = sp.cho_cov_ylm.eval()
    y = mu[:, None] + np.dot(cho_cov, np.random.randn(256, nsamples))

    # Draw sin-distributed inclinations
    inc = np.arccos(np.random.random(nsamples)) * 180 / np.pi

    # Compute all the light curves
    map = starry.Map(sp._ydeg)
    f = np.zeros((nsamples, len(t)))
    for k in tqdm(range(nsamples)):
        map.inc = inc[k]
        A = map.design_matrix(theta=360 * t)
        f[k] = np.transpose(A @ y[:, k])

    # Return the empirical mean and covariance
    return np.mean(f, axis=0), np.cov(f.T)


def test_inclination(nsamples=10000, plot=False, rtol=1e-4, ftol=0.25):
    """
    Test the inclination marginalization algorithm.

    """
    # Time array
    t = np.linspace(0, 1, 1000)

    # Compute the analytic moments
    sp = StarryProcess(normalized=False, marginalize_over_inclination=True)
    mean = sp.mean(t).eval()
    cov = sp.cov(t).eval()

    # Compute the numerical moments
    np.random.seed(0)
    mean_num, cov_num = get_numerical_mean_and_cov(t, nsamples=nsamples)

    # Visualize
    if plot:

        # The radial kernel
        plt.figure()
        plt.plot(cov[0])
        plt.plot(cov_num[0])

        # The full covariance
        fig, ax = plt.subplots(1, 3)
        vmin = np.min(cov)
        vmax = np.max(cov)
        im = ax[0].imshow(cov, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(cov_num, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(np.log10(np.abs((cov - cov_num) / cov)))
        plt.colorbar(im, ax=ax[2])
        plt.show()

    # Check
    rerr = np.abs(cov[0] - cov_num[0])
    assert np.max(rerr) < rtol, "relative error too large"

    ferr = np.abs((cov[0] - cov_num[0]) / cov[0, 0])
    assert np.max(ferr) < ftol, "fractional error too large"


if __name__ == "__main__":
    starry.config.lazy = False
    test_inclination(plot=True)
