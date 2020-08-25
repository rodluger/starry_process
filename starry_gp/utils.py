import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import legendre
from packaging import version
import logging

# Set up the logger
logger = logging.getLogger("starry_process")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Kewyord to `eigh` changed in 1.5.0
if version.parse(scipy.__version__) < version.parse("1.5.0"):
    eigvals = "eigvals"
    driver_allowed = False
else:
    eigvals = "subset_by_index"
    driver_allowed = True


__all__ = ["eigen", "get_alpha_beta"]


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
