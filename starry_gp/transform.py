from .wigner import R
import numpy as np
import scipy
from scipy.linalg import eigh
from packaging import version

# Kewyord to `eigh` changed in 1.5.0
if version.parse(scipy.__version__) < version.parse("1.5.0"):
    eigvals = "eigvals"
else:
    eigvals = "subset_by_index"


def eigen(Q, n=None):
    N = Q.shape[0]
    if n is None:
        n = N
    w, U = eigh(Q, **{eigvals: (N - n, N - 1)})
    U = U @ np.diag(np.sqrt(np.maximum(0, w)))
    return U[:, ::-1]


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
        self.T = [np.zeros((self.n, self.n, self.n)) for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            self.T[l] = np.swapaxes(self.R[l] @ self.U[i], 1, 2)

    def set_params(self, *args, **kwargs):
        self._compute_basis_integrals(*args, **kwargs)
        self._compute_U()
        self._compute_t()
        self._compute_T()

    def first_moment(self, s):
        """Compute the first moment of the distribution."""
        mu = np.zeros(self.N)
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            mu[i] = self.t[l] @ s[i]
        return mu

    def second_moment(self, sqrtS):
        """Compute the second moment of the distribution."""
        sqrtC = np.zeros((self.N, self.n, sqrtS.shape[-1]))
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            sqrtC[i] = self.T[l] @ sqrtS[i]
        return sqrtC.reshape(self.N, -1)
