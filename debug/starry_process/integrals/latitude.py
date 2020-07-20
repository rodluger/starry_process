import numpy as np
from scipy.special import hyp2f1, gamma
from scipy.integrate import quad
from scipy.stats import beta as Beta
from scipy.linalg import block_diag
from . import wigner


class LatitudeIntegral(object):
    def __init__(self, ydeg, skip_nullspace=False):
        self.ydeg = ydeg
        self.ls = np.arange(self.ydeg + 1)
        if skip_nullspace:
            self.ls = np.append(self.ls[:3], self.ls[4::2])
        self.N = (ydeg + 1) ** 2
        self._R = wigner.R(ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1)
        self.set_params(2, 2)
        self.set_matrix(np.eye(self.N))
        self.set_vector(np.ones(self.N))

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

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

    def set_params(self, alpha, beta):
        # Store these
        self._alpha = alpha
        self._beta = beta
        self._BetaNorm = gamma(alpha) * gamma(beta) / gamma(alpha + beta)

        # B functions
        B = np.zeros(4 * self.ydeg + 1)
        B[0] = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
        for n in range(1, 4 * self.ydeg + 1):
            B[n] = (alpha - 1 + n) / (alpha + beta - 1 + n) * B[n - 1]

        # F functions
        ab = alpha + beta
        F = np.zeros(4 * self.ydeg + 1)
        F[0] = hyp2f1(-0.5, alpha, ab, -1)
        F[1] = hyp2f1(-0.5, alpha + 1, ab + 1, -1)
        for n in range(2, 4 * self.ydeg + 1):
            F[n] = (
                (ab + n - 1)
                / ((alpha + n - 1) * (ab + n - 0.5))
                * ((ab + n - 2) * F[n - 2] + (1.5 - beta) * F[n - 1])
            )
        F *= B

        # Terms
        self._term = np.zeros((4 * self.ydeg + 1, 4 * self.ydeg + 1))
        for i in range(4 * self.ydeg + 1):
            if i % 2 == 0:
                func = B
            else:
                func = F
            i2 = i // 2
            for j in range(0, 4 * self.ydeg + 1, 2):
                j2 = j // 2
                fac1 = 1.0
                for k1 in range(i2 + 1):
                    fac2 = fac1
                    for k2 in range(j2 + 1):
                        self._term[i, j] += fac2 * func[k1 + k2]
                        fac2 *= (k2 - j2) / (k2 + 1.0)
                    fac1 *= (i2 - k1) / (k1 + 1)

    def _test_term(self):
        def num_integrand(phi, i, j):
            x = np.cos(phi)
            jac = 0.5 * np.abs(np.sin(phi))
            return (
                (1 + x) ** (0.5 * i)
                * (1 - x) ** (0.5 * j)
                * x ** (self.alpha - 1)
                * (1 - x) ** (self.beta - 1)
                * np.sign(phi) ** j
                * jac
            )

        for i in range(4 * self.ydeg + 1):
            for j in range(4 * self.ydeg + 1):
                result = quad(num_integrand, -0.5 * np.pi, 0.5 * np.pi, (i, j))[0]
                assert np.allclose(result, self._term[i, j])

    def integral1(self, vector=None, alpha=None, beta=None):
        if vector is not None:
            self.set_vector(vector)
        if alpha is not None or beta is not None:
            self.set_params(alpha, beta)
        P = np.zeros(self.N)
        for l in self.ls:
            term = np.diag(np.fliplr(self._term[: 2 * l + 1, : 2 * l + 1]))
            i = slice(l ** 2, (l + 1) ** 2)
            P[i] = 1.0 / (2 ** l * self._BetaNorm) * self._q[i, : 2 * l + 1].dot(term)
        return P

    def _test_integral1(self):
        np.random.seed(1)
        vector = np.random.randn(self.N)
        ls = np.array(self.ls)
        self.ls = np.arange(self.ydeg + 1)
        P = self.integral1(vector=vector)
        self.ls = ls
        Pnum = np.zeros_like(P)
        phi_arr = np.linspace(-np.pi / 2, np.pi / 2, 10000)
        dphi = phi_arr[1] - phi_arr[0]
        for phi in phi_arr:
            R = wigner._R(
                self.ydeg, phi, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
            )
            Rs = block_diag(*R).dot(vector)
            jac = 0.5 * np.abs(np.sin(phi))
            Pnum += Rs * jac * Beta.pdf(np.cos(phi), self.alpha, self.beta) * dphi
        assert np.allclose(Pnum, P, atol=1e-3)

    def integral2(self, matrix=None, alpha=None, beta=None):
        if matrix is not None:
            self.set_matrix(matrix)
        if alpha is not None or beta is not None:
            self.set_params(alpha, beta)
        P = np.zeros((self.N, self.N))
        for l1 in self.ls:
            i = slice(l1 ** 2, (l1 + 1) ** 2)
            for l2 in self.ls:
                j = slice(l2 ** 2, (l2 + 1) ** 2)
                l = l1 + l2
                term = np.diag(np.fliplr(self._term[: 2 * l + 1, : 2 * l + 1]))
                P[i, j] = (
                    1.0
                    / (2 ** l * self._BetaNorm)
                    * self._Q[i, j, : 2 * l + 1].dot(term)
                )
        return P

    def _test_integral2(self):
        np.random.seed(1)
        matrix = np.tril(np.random.randn(self.N, self.N))
        matrix = np.dot(matrix, matrix.T)
        ls = np.array(self.ls)
        self.ls = np.arange(self.ydeg + 1)
        P = self.integral2(matrix=matrix)
        self.ls = ls
        Pnum = np.zeros_like(P)
        phi_arr = np.linspace(-np.pi / 2, np.pi / 2, 10000)
        dphi = phi_arr[1] - phi_arr[0]
        for phi in phi_arr:
            R = wigner._R(
                self.ydeg, phi, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
            )
            RRT = block_diag(*R).dot(matrix).dot(block_diag(*R).T)
            jac = 0.5 * np.abs(np.sin(phi))
            Pnum += RRT * jac * Beta.pdf(np.cos(phi), self.alpha, self.beta) * dphi

        assert np.allclose(Pnum, P)


def test():
    integral = LatitudeIntegral(4)
    integral._test_term()
    integral._test_integral1()
    integral._test_integral2()
