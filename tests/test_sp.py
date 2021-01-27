from starry_process.sp import StarryProcess
from starry_process.wigner import R
from starry_process.defaults import defaults
from starry_process.size import Spot
import numpy as np
from tqdm import tqdm


def test_moments_by_sampling(rtol=1e-3, ftol=3e-2):

    # Settings
    ydeg = defaults["ydeg"]
    ydeg_num = 5
    np.random.seed(0)
    nsamples = int(1e5)
    r = defaults["r"]
    a = defaults["a"]
    b = defaults["b"]
    c = defaults["c"]
    n = defaults["n"]

    # Integrate analytically
    print("Computing moments analytically...")
    gp = StarryProcess(ydeg=ydeg, r=r, a=a, b=b, c=c, n=n)
    mu = gp.mean_ylm.eval()
    cov = gp.cov_ylm.eval()

    # Integrate numerically
    print("Computing moments numerically...")

    # Draw the latitude and longitude
    phi = gp.latitude.sample(nsamples=nsamples).eval()
    lam = gp.longitude.sample(nsamples=nsamples).eval()

    # Compute the spot expansion at (0, 0)
    s = Spot(ydeg=ydeg).get_y(r * np.pi / 180).eval()

    # Integrate numerically by sampling. We'll only
    # compute things up to `ydeg_num` since this is *very*
    # expensive to do!
    nylm = (ydeg_num + 1) ** 2
    y = np.empty((nsamples, nylm))
    for k in tqdm(range(nsamples)):

        # Rotation in latitude
        Rx = R(
            ydeg_num,
            phi=phi[k] * np.pi / 180.0,
            cos_alpha=0,
            sin_alpha=1,
            cos_gamma=0,
            sin_gamma=-1,
        )

        # Rotation in longitude
        Ry = R(
            ydeg_num,
            phi=lam[k] * np.pi / 180.0,
            cos_alpha=1,
            sin_alpha=0,
            cos_gamma=1,
            sin_gamma=0,
        )

        # Apply the transformations
        for l in range(ydeg_num + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            y[k, i] = Ry[l] @ (Rx[l] @ s[i])

    mu_num = np.pi * c * n * np.mean(y, axis=0)
    cov_num = (np.pi * c) ** 2 * n * np.cov(y.T)

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
