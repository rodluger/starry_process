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
        w, self.U = eigh(self.Q, **{eigvals: (self.N - self.n, self.N - 1)})
        self.U = self.U @ np.diag(np.sqrt(np.maximum(0, w)))
        self.U = self.U[:, ::-1]

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
        sqrtC = np.zeros((self.N, self.n, self.N))
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            sqrtC[i] = self.T[l] @ sqrtS[i]
        A = sqrtC.reshape(self.N, -1)
        C = A @ A.T
        return C
