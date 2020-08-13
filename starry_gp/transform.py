from .wigner import R
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.optimize import minimize
from packaging import version

# Kewyord to `eigh` changed in 1.5.0
if version.parse(scipy.__version__) < version.parse("1.5.0"):
    eigvals = "eigvals"
else:
    eigvals = "subset_by_index"


__all__ = ["eigen", "get_c0_c1", "get_alpha_beta", "TransformIntegral"]


def eigen(Q, n=None):
    """
    
    """
    N = Q.shape[0]
    if n is None:
        n = N
    w, U = eigh(Q, **{eigvals: (N - n, N - 1)})
    U = U @ np.diag(np.sqrt(np.maximum(0, w)))
    return U[:, ::-1]


def hwhm(rprime):
    """
    Return the half-width at half minimum as a function of r'.
    
    """
    return (
        np.arccos((2 + 3 * rprime * (2 + rprime)) / (2 * (1 + rprime) ** 3))
        * 180
        / np.pi
    )


def peak_error(ydeg, rprime):
    """
    Returns the error in the intensity at the spot center.
    
    """
    xi = 1.0
    I = 1 - 0.5 * xi * rprime / (1 + rprime)
    for l in range(1, ydeg + 1):
        I -= 0.5 * xi * rprime * (2 + rprime) / (1 + rprime) ** (l + 1)
    return np.abs(I)


def min_rprime(ydeg, tol=1e-2):
    """
    Returns the smallest value of r' such that the error on the peak
    intensity is less than `tol`.

    """
    f = lambda rprime: (peak_error(ydeg, rprime) - tol) ** 2
    res = minimize(f, 0.25)
    return res.x


def max_rprime(hmwhm_max=75):
    """
    Returns the value of r' corresponding to `hwhm_max`.
    
    """
    f = lambda rprime: (hwhm(rprime) - hmwhm_max) ** 2
    res = minimize(f, 10)
    return res.x


def get_c0_c1(ydeg, tol=1e-2, hwhm_max=75):
    """
    Returns the c_0, c_1 coefficients for the radius transformation.

    """
    rmin = min_rprime(ydeg, tol=tol)
    rmax = max_rprime(hmwhm_max=hwhm_max)
    c0 = rmin
    c1 = rmax - rmin
    return c0, c1


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


class TransformIntegral(object):
    def __init__(self, ydeg, **wigner_kwargs):
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2
        self.n = 2 * ydeg + 1

        # Compute the Wigner matrices
        self.R = R(ydeg, **wigner_kwargs)

    def _compute_basis_integrals(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _compute_U(self):
        self.U = eigen(self.Q, self.n)

    def _compute_t(self):
        self.t = [np.zeros((self.n, self.n)) for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            self.t[l] = self.R[l] @ self.q[l ** 2 : (l + 1) ** 2]

    def _compute_T(self):
        self.T = [
            np.zeros((self.n, self.n, self.n)) for l in range(self.ydeg + 1)
        ]
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            self.T[l] = np.swapaxes(self.R[l] @ self.U[i], 1, 2)

    def set_params(self, *args, **kwargs):
        self._compute_basis_integrals(*args, **kwargs)
        self._compute_U()
        self._compute_t()
        self._compute_T()

    def first_moment(self, e):
        """Compute the first moment of the distribution."""
        mu = np.zeros(self.N)
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            mu[i] = self.t[l] @ e[i]
        return mu

    def second_moment(self, eigE):
        """Compute the second moment of the distribution."""
        sqrtC = np.zeros((self.N, self.n, eigE.shape[-1]))
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            sqrtC[i] = self.T[l] @ eigE[i]
        return sqrtC.reshape(self.N, -1)
