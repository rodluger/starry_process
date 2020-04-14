import numpy as np
from scipy.special import hyp2f1, gamma
from scipy.integrate import quad
from scipy.stats import beta as Beta
from scipy.linalg import block_diag
from . import wigner


class LongitudeIntegral(object):
    def __init__(self, ydeg, skip_nullspace=False):
        self.ydeg = ydeg
        self.ls = np.arange(self.ydeg + 1)
        if skip_nullspace:
            self.ls = np.append(self.ls[:3], self.ls[4::2])
        self.N = (ydeg + 1) ** 2
        self._R = wigner.R(ydeg, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0)
        self.set_matrix(np.eye(self.N))
        self.set_vector(np.ones(self.N))
        self._precompute()

    @property
    def matrix(self):
        return self._matrix

    def set_matrix(self, matrix):
        self._matrix = matrix
        self._Q = wigner.MatrixDot(self._R, self._matrix)

    @property
    def vector(self):
        return self._vector

    def set_vector(self, vector):
        self._vector = vector
        self._q = wigner.VectorDot(self._R, self._vector)

    def _precompute(self):
        # TODO: Check for stability here.
        self._term = np.zeros((4 * self.ydeg + 1, 4 * self.ydeg + 1))
        for i in range(4 * self.ydeg + 1):
            for j in range(0, 4 * self.ydeg + 1, 2):
                self._term[i, j] = (
                    2
                    * gamma(0.5 * (i + 1))
                    * gamma(0.5 * (j + 1))
                    / gamma(0.5 * (2 + i + j))
                )

    def _test_term(self):
        def num_integrand(lam, i, j):
            return np.cos(lam / 2) ** i * np.sin(lam / 2) ** j

        for i in range(2 * self.ydeg + 1):
            for j in range(2 * self.ydeg + 1):
                result = quad(num_integrand, -np.pi, np.pi, (i, j))[0]
                assert np.allclose(result, self._term[i, j])

    def integral1(self, vector=None):
        """
        Return the integral of the first moment.

        """
        if vector is not None:
            self.set_vector(vector)
        L = np.zeros(self.N)
        for l in self.ls:
            term = np.diag(np.fliplr(self._term[: 2 * l + 1, : 2 * l + 1]))
            i = slice(l ** 2, (l + 1) ** 2)
            L[i] = 1.0 / (2 * np.pi) * self._q[i, : 2 * l + 1].dot(term)
        return L

    def _test_integral1(self):
        np.random.seed(1)
        vector = np.random.randn(self.N)
        ls = np.array(self.ls)
        self.ls = np.arange(self.ydeg + 1)
        L = self.integral1(vector=vector)
        self.ls = ls
        Lnum = np.zeros_like(L)
        lam_arr = np.linspace(-np.pi, np.pi, 10000)
        dlam = lam_arr[1] - lam_arr[0]
        for lam in lam_arr:
            R = wigner._R(
                self.ydeg, lam, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0
            )
            Rs = block_diag(*R).dot(vector)
            Lnum += Rs * dlam / (2 * np.pi)
        assert np.allclose(Lnum, L, atol=1e-3)

    def integral2(self, matrix=None):
        """
        Return the integral of the second moment.

        """
        if matrix is not None:
            self.set_matrix(matrix)
        L = np.zeros((self.N, self.N))
        for l1 in self.ls:
            i = slice(l1 ** 2, (l1 + 1) ** 2)
            for l2 in self.ls:
                j = slice(l2 ** 2, (l2 + 1) ** 2)
                l = l1 + l2
                term = np.diag(np.fliplr(self._term[: 2 * l + 1, : 2 * l + 1]))
                L[i, j] = 1.0 / (2 * np.pi) * self._Q[i, j, : 2 * l + 1].dot(term)
        return L

    def _test_integral2(self):
        np.random.seed(1)
        matrix = np.tril(np.random.randn(self.N, self.N))
        matrix = np.dot(matrix, matrix.T)
        ls = np.array(self.ls)
        self.ls = np.arange(self.ydeg + 1)
        L = self.integral2(matrix=matrix)
        self.ls = ls
        Lnum = np.zeros_like(L)
        lam_arr = np.linspace(-np.pi, np.pi, 10000)
        dlam = lam_arr[1] - lam_arr[0]
        for lam in lam_arr:
            R = wigner._R(
                self.ydeg, lam, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0
            )
            RPRT = block_diag(*R).dot(matrix).dot(block_diag(*R).T)
            Lnum += RPRT * dlam / (2 * np.pi)
        assert np.allclose(Lnum, L, atol=1e-2)


def test():
    integral = LongitudeIntegral(4)
    integral._test_term()
    integral._test_integral1()
    integral._test_integral2()
