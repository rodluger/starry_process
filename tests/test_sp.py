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


def test_moments_by_integration(rtol=5e-3, ftol=1e-2):

    # Settings
    ydeg = 15
    ydeg_num = 5
    atol = 1e-4
    np.random.seed(0)
    nsamples = int(1e5)

    # Integrate analytically
    print("Computing moments analytically...")
    gp = StarryProcess(ydeg, **defaults)
    mu = gp.mean_ylm.eval()
    cov = gp.cov_ylm.eval()

    # Integrate numerically
    print("Computing moments numerically...")

    # Draw the size, contrast, latitude, and amplitude
    hwhm = gp.size.transform.sample(
        a=defaults["sa"], b=defaults["sb"], nsamples=nsamples
    )
    xi = gp.contrast.transform.sample(
        defaults["ca"], defaults["cb"], nsamples=nsamples
    )
    phi = gp.latitude.transform.sample(
        a=defaults["la"], b=defaults["lb"], nsamples=nsamples
    )
    lam = gp.longitude.transform.sample(nsamples=nsamples)

    # Integrate numerically by sampling. We'll only
    # compute things up to `ydeg_num` since this is *very*
    # expensive to do!
    N = (ydeg_num + 1) ** 2
    y = np.empty((nsamples, N))
    for n in tqdm(range(nsamples)):

        # Compute the spot expansion at (0, 0)
        s = gp.size.transform.get_s(hwhm=hwhm[n]).reshape(-1)[:N]

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
            y[n, i] = xi[n] * (Ry[l] @ (Rx[l] @ s[i]))

    mu_num = np.mean(y, axis=0)
    cov_num = np.cov(y.T)

    # Avoid div by zero in the comparison
    nonzero_i = np.abs(mu[:N]) > 1e-4
    nonzero_ij = np.abs(cov[:N, :N]) > 1e-4

    # Compare
    assert np.max(np.abs(mu[:N] - mu_num)) < rtol, "error in mean"
    assert (
        np.max(np.abs(1 - mu[:N][nonzero_i] / mu_num[nonzero_i])) < ftol
    ), "error in mean"
    assert np.max(np.abs(cov[:N, :N] - cov_num)) < rtol, "error in cov"
    assert (
        np.max(np.abs(1 - cov[:N, :N][nonzero_ij] / cov_num[nonzero_ij]))
        < ftol
    ), "error in cov"


def test_moments_by_sampling(plot=False):

    la = 0.5
    lb = 0.5
    s = 20.0
    c = 0.5
    N = 20

    # Generate from the true distribution
    nsamples = 1000
    np.random.seed(0)
    sp = StarryProcess()
    ydeg = 15
    map = starry.Map(ydeg, lazy=False)
    y0 = c * sp.size.spot.get_y(s * np.pi / 180).eval()
    y_true = np.zeros((nsamples, (ydeg + 1) ** 2))
    for k in tqdm(range(nsamples)):
        for n in range(N):
            lat = sp.latitude.transform.sample(a=la, b=lb)
            lon = np.random.uniform(-180, 180)
            map[:, :] = y0
            map.rotate([1, 0, 0], lat)
            map.rotate([0, 1, 0], lon)
            y_true[k] += map.amp * map.y
    mean_true = np.mean(y_true, axis=0)
    std_true = np.std(y_true, axis=0)

    # Sample from the GP
    sp = StarryProcess(s=s, la=la, lb=lb, c=c, N=N)
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


def test_sample():

    # Settings
    ydeg = 15
    t = np.linspace(0, 1, 500)

    # Compute
    gp = StarryProcess(ydeg, **defaults)
    fluxes = gp.sample(t=t, nsamples=10).eval()


@pytest.mark.parametrize("param", ["la", "lb", "sa", "sb"])
def test_grad(
    param, abs_tol=1e-5, rel_tol=1e-5, eps=1e-6,
):

    with change_flags(compute_test_value="off"):

        np.random.seed(0)
        npts = 1000
        t = np.linspace(0, 1, npts)
        flux = np.random.randn(npts)

        def func(x):
            kwargs = {param: x}
            return StarryProcess(**kwargs, driver="numpy").log_likelihood(
                t, flux, 1.0
            )

        x = defaults[param]

        verify_grad(
            func, (x,), n_tests=1, abs_tol=abs_tol, rel_tol=rel_tol, eps=eps,
        )


if __name__ == "__main__":
    test_moments_by_sampling(plot=True)
