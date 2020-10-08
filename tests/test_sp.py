from starry_process.sp import StarryProcess
from starry_process.wigner import R
from starry_process.defaults import defaults
from starry_process.size import DiscreteSpot
import numpy as np
from tqdm import tqdm
from scipy.stats import beta as Beta
from scipy.stats import lognorm as LogNormal
import theano
from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags
import theano.tensor as tt
import pytest
import starry
import matplotlib.pyplot as plt


def test_moments_by_integration(rtol=3e-2, ftol=3e-2):

    # Settings
    ydeg = 15
    ydeg_num = 5
    atol = 1e-4
    np.random.seed(0)
    nsamples = int(1e5)
    sa = 0.5
    sb = 0.5
    la = 0.5
    lb = 0.5
    c = 0.1
    N = 3.0

    # Integrate analytically
    print("Computing moments analytically...")
    gp = StarryProcess(
        ydeg=ydeg, size=[sa, sb], latitue=[la, lb], contrast=[c, N]
    )
    mu = gp.mean_ylm.eval()
    cov = gp.cov_ylm.eval()

    # Integrate numerically
    print("Computing moments numerically...")

    # Draw the size, contrast, latitude, and amplitude
    hwhm = gp.size.sample(nsamples=nsamples).eval()
    phi = gp.latitude.sample(nsamples=nsamples).eval()
    lam = gp.longitude.sample(nsamples=nsamples).eval()

    # Integrate numerically by sampling. We'll only
    # compute things up to `ydeg_num` since this is *very*
    # expensive to do!
    nylm = (ydeg_num + 1) ** 2
    y = np.empty((nsamples, nylm))
    for n in tqdm(range(nsamples)):

        # Compute the spot expansion at (0, 0)
        s = gp.size._transform.get_s(hwhm=hwhm[n]).reshape(-1)[:nylm]

        # Rotation in latitude
        Rx = R(
            ydeg_num,
            phi=phi[n] * np.pi / 180.0,
            cos_alpha=0,
            sin_alpha=1,
            cos_gamma=0,
            sin_gamma=-1,
        )

        # Rotation in longitude
        Ry = R(
            ydeg_num,
            phi=lam[n] * np.pi / 180.0,
            cos_alpha=1,
            sin_alpha=0,
            cos_gamma=1,
            sin_gamma=0,
        )

        # Apply the transformations
        for l in range(ydeg_num + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            y[n, i] = np.pi * c * (Ry[l] @ (Rx[l] @ s[i]))

    mu_num = N * np.mean(y, axis=0)
    cov_num = N * np.cov(y.T)

    breakpoint()

    # Avoid div by zero in the comparison
    nonzero_i = np.abs(mu[:nylm]) > 1e-4
    nonzero_ij = np.abs(cov[:nylm, :nylm]) > 1e-4

    # Compare
    assert np.max(np.abs(mu[:nylm] - mu_num)) < rtol, "error in mean"
    assert (
        np.max(np.abs(1 - mu[:nylm][nonzero_i] / mu_num[nonzero_i])) < ftol
    ), "error in mean"
    assert np.max(np.abs(cov[:nylm, :nylm] - cov_num)) < rtol, "error in cov"
    assert (
        np.max(np.abs(1 - cov[:nylm, :nylm][nonzero_ij] / cov_num[nonzero_ij]))
        < ftol
    ), "error in cov"


def test_moments_by_sampling(plot=False):

    la = 0.5
    lb = 0.5
    s = 20.0
    c = 0.5
    N = 20

    # Sample from the true distribution, then
    # compute empirical mean and std dev
    nsamples = 10000
    np.random.seed(0)
    sp = StarryProcess(size=s, latitude=[la, lb], contrast=[c, N])
    ydeg = 15
    map = starry.Map(ydeg)
    y0 = np.pi * c * sp.size._spot.get_y(s * np.pi / 180).eval()
    y_true = np.zeros((nsamples, (ydeg + 1) ** 2))
    lat = sp.latitude.sample(nsamples=nsamples * N).eval().reshape(nsamples, N)
    lon = (
        sp.longitude.sample(nsamples=nsamples * N).eval().reshape(nsamples, N)
    )
    for k in tqdm(range(nsamples)):
        for n in range(N):
            map[:, :] = y0
            map.rotate([1, 0, 0], lat[k, n])
            map.rotate([0, 1, 0], lon[k, n])
            y_true[k] += map.amp * map.y
    mean_true = np.mean(y_true, axis=0)
    std_true = np.std(y_true, axis=0)

    # Compute the exact mean and std dev
    mean_gp = sp.mean_ylm.eval()
    std_gp = np.sqrt(np.diag(sp.cov_ylm.eval()))

    # Compute the ratio
    mean_ratio = mean_true / mean_gp
    mean_ratio[np.abs(mean_gp) < 1e-5] = np.nan
    std_ratio = std_true / std_gp
    std_ratio[np.abs(std_gp) < 1e-5] = np.nan

    # Plot
    if plot:

        fig, ax = plt.subplots(3, sharex=True)
        ax[0].plot(mean_true, label="true")
        ax[0].plot(mean_gp, label="gp")
        ax[0].set_ylabel("mean")
        ax[0].legend()

        ax[1].plot(std_true, label="true")
        ax[1].plot(std_gp, label="gp")
        ax[1].set_ylabel("std")
        ax[1].legend()

        ax[2].plot(mean_ratio, ".", label="mean")
        ax[2].plot(std_ratio, label="std")
        ax[2].set_ylabel("true / gp")
        ax[2].set_xlabel("ylm index")
        ax[2].legend()

        for axis in ax:
            for l in range(ydeg + 1):
                axis.axvline(l * (l + 1), color="k", lw=1, alpha=0.5, ls="--")

        plt.show()

    assert np.abs(np.nanmean(mean_ratio) - 1) < 1e-2
    assert np.abs(np.nanmean(std_ratio) - 1) < 1e-2


if __name__ == "__main__":
    starry.config.lazy = False
    test_moments_by_sampling(plot=True)
