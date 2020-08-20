import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import legendre
from packaging import version

# Kewyord to `eigh` changed in 1.5.0
if version.parse(scipy.__version__) < version.parse("1.5.0"):
    eigvals = "eigvals"
    driver_allowed = False
else:
    eigvals = "subset_by_index"
    driver_allowed = True


__all__ = ["eigen", "get_alpha_beta", "get_c"]


def eigen(Q, n=None, driver=None):
    """
    Returns the matrix square root of `Q`,
    computed via (hermitian) eigendecomposition:

        eigen(Q) . eigen(Q)^T = Q

    """
    N = Q.shape[0]
    if n is None or n == N:
        if driver_allowed:
            kwargs = {"driver": driver}
        else:
            kwargs = {}
    else:
        kwargs = {eigvals: (N - n, N - 1)}
    w, U = eigh(Q, **kwargs)
    U = U @ np.diag(np.sqrt(np.maximum(0, w)))
    return U[:, ::-1]


def get_alpha_beta(mu, nu):
    """
    Compute the parameters `alpha` and `beta` of a Beta distribution,
    given its mean `mu` and normalized variance `nu`.

    The mean `mu` is the mean of the Beta distribution, valid in (0, 1).
    The normalized variance `nu` is the variance of the Beta distribution
    divided by `mu * (1 - mu)`, valid in `(0, 1)`.

    """
    assert np.all(mu > 0) and np.all(mu < 1), "mean must be in (0, 1)."
    assert np.all(nu > 0) and np.all(nu < 1), "variance must be in (0, 1)."
    alpha = mu * (1 / nu - 1)
    beta = (1 - mu) * (1 / nu - 1)
    return alpha, beta


def hwhm(rhop):
    """
    Return the theoretical half-witdth at half minimum as a function of rho'.
    
    """
    return (
        np.arccos((2 + 3 * rhop * (2 + rhop)) / (2 * (1 + rhop) ** 3))
        * 180
        / np.pi
    )


def hwhm_inv(hwhm):
    """
    Return rho' as a function of the theoretical half-width at half minimum.
    
    """
    theta = hwhm * np.pi / 180
    return (1 + np.cos(2 * theta / 3) + np.sqrt(3) * np.sin(2 * theta / 3)) / (
        2 * np.cos(theta)
    ) - 1


def rhop_max(hwhm_max=60):
    """
    Returns the value of rho' corresponding to `hwhm_max`.
    
    """
    f = lambda rhop: (hwhm(rhop) - hwhm_max) ** 2
    res = minimize(f, 2.0)
    return res.x[0]


def corr(rhop, c):
    """Intensity correction function."""
    rho = (rhop - c[0]) / c[1]
    return 1 + c[2] * (1 - rho) ** c[3]


def I(ydeg, rhop, theta, c=None):
    """
    Return the intensity at polar angle `theta` (in deg) away from
    the center of a spot of radius rho' expanded to degree `ydeg`.
    
    """
    # Compute the Legendre expansion
    cost = np.cos(theta * np.pi / 180)
    term = np.sum(
        [(1 + rhop) ** -l * legendre(l)(cost) for l in range(ydeg + 1)], axis=0
    )
    I = 0.5 * rhop * (1 - (2 + rhop) / (1 + rhop) * term)

    # Apply the intensity correction
    if c is not None:
        I *= corr(rhop, c)

    return I


def get_c(ydeg, hwhm_max=75, hwhm_min=15, npts=500):
    """
    Return the coefficients for the radius transformation.

    """
    c = np.zeros(4)

    # Minimum r: we need to optimize numerically
    loss = lambda p: (hwhm_empirical(ydeg, p[0]) - hwhm_min) ** 2
    res = minimize(loss, hwhm_inv(hwhm_min))
    rhopmin = res.x[0]
    c[0] = rhopmin

    # Maximum r (easy)
    rhopmax = rhop_max(hwhm_max=hwhm_max)
    c[1] = rhopmax - rhopmin

    # Now compute the coefficients of the intensity
    # correction, c[2] and c[3].

    # Array over which to compute the loss
    rhop = np.linspace(rhopmin + 1e-6, rhopmax - 1e-6, npts)

    # Get the actual (absolute value of the) intensity at the peak
    l = np.arange(ydeg + 1).reshape(1, -1)
    term = np.sum((1 + rhop.reshape(-1, 1)) ** -l, axis=-1)
    I = -0.5 * rhop * (1 - (2 + rhop) / (1 + rhop) * term)

    # This is the factor by which we need to normalize the function
    norm = 1.0 / I

    # Find the coefficients of the fit (least squares)
    diff = lambda p: np.sum((norm - corr(rhop, [c[0], c[1], p[0], p[1]])) ** 2)
    res = minimize(diff, [0.1, 50.0])
    c[2:] = res.x

    # TODO: Use logging
    print(
        "Delta theta range: {:.2f} - {:.2f} degrees".format(hwhm_min, hwhm_max)
    )
    print("c coeffs: {:s}".format(" ".join(["{:.8f}".format(ck) for ck in c])))
    print(
        "Maximum intensity |error|: {:.2e}".format(
            np.max(np.abs(norm - corr(rhop, c)))
        )
    )
    print(
        "Average intensity |error|: {:.2e}".format(
            np.mean(np.abs(norm - corr(rhop, c)))
        )
    )

    return c


def hwhm_empirical(ydeg, rhop):
    """
    Return the empirical half-width at half minimum as a function of rho'.
    
    """
    # Setup
    rhop = np.atleast_1d(rhop)
    hwhm_empirical = np.zeros_like(rhop)

    # Find the HWHM numerically for each radius
    for k in range(len(rhop)):

        halfmax = 0.5 * I(ydeg, rhop[k], 0)

        def loss(theta):
            return (I(ydeg, rhop[k], theta) - halfmax) ** 2

        res = minimize(loss, hwhm(max(0.1, rhop[k])))
        hwhm_empirical[k] = res.x[0]

    return hwhm_empirical
