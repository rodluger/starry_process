from starry_process.sp import StarryProcess
from starry_process.wigner import R
from starry_process.defaults import defaults
import numpy as np
from tqdm import tqdm
from scipy.stats import beta as Beta
from scipy.stats import lognorm as LogNormal


def test_moments(rtol=5e-3, ftol=1e-2):

    # Settings
    ydeg = 15
    ydeg_num = 5
    atol = 1e-4
    np.random.seed(0)
    nsamples = int(1e5)

    # Integrate analytically
    print("Computing moments analytically...")
    gp = StarryProcess(ydeg, **defaults)
    mu = gp.mean_ylm().eval()
    cov = gp.cov_ylm().eval()

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


def test_sample():

    # Settings
    ydeg = 15
    t = np.linspace(0, 1, 500)

    # Compute
    gp = StarryProcess(ydeg, **defaults)
    fluxes = gp.sample(t=t, nsamples=10).eval()
