import numpy as np
from scipy.special import hyp2f1
from scipy.special import beta as EulerBeta


class SizeIntegralOp(object):
    def __init__(self, ydeg, **kwargs):
        self.ydeg = ydeg
        self.N = (self.ydeg + 1) ** 2
        self._IMAX = 3
        self._KMAX = 2
        self._MMAX = 4
        self._JMAX = 2 * self.ydeg + 1
        self._c = np.array(
            [float(kwargs["SP__C{0}".format(n)]) for n in range(4)]
        )
        self._W = np.array(
            [
                self._c[0],
                self._c[0],
                0,
                0,
                0,
                (self._c[0] * self._c[0]),
                (2 * self._c[0] * self._c[0]),
                (self._c[0] * self._c[0]),
                0,
                0,
                (self._c[0] * self._c[0] * self._c[0]),
                (3 * self._c[0] * self._c[0] * self._c[0]),
                (3 * self._c[0] * self._c[0] * self._c[0]),
                (self._c[0] * self._c[0] * self._c[0]),
                0,
                (self._c[0] * self._c[0] * self._c[0] * self._c[0]),
                (4 * self._c[0] * self._c[0] * self._c[0] * self._c[0]),
                (6 * self._c[0] * self._c[0] * self._c[0] * self._c[0]),
                (4 * self._c[0] * self._c[0] * self._c[0] * self._c[0]),
                (self._c[0] * self._c[0] * self._c[0] * self._c[0]),
            ]
        ).reshape(self._IMAX + 1, self._MMAX + 1)

    def _getGfac(self, alpha, beta):
        lam = np.zeros((self._KMAX + 1, self._MMAX + 1))
        lam[:, 0] = 1
        q = self._c[1] / self._c[0]
        for m in range(1, self._MMAX + 1):
            for k in range(self._KMAX + 1):
                lam[k, m] = (
                    lam[k, m - 1]
                    * q
                    * (alpha + m - 1)
                    / (alpha + beta + k * self._c[3] + m - 1)
                )
        norm = self._c[2] / EulerBeta(alpha, beta)
        for k in range(1, self._KMAX + 1):
            lam[k] *= norm * EulerBeta(alpha, beta + k * self._c[3])
            norm *= self._c[2]
        return lam

    def _G_num(self, alpha, beta, j, k, m):
        z = -self._c[1] / (1.0 + self._c[0])
        p = 1.0 + self._c[0] + self._c[1]
        zbar = self._c[1] / p
        G = hyp2f1(
            1.0 + j,
            beta + k * self._c[3],
            alpha + beta + k * self._c[3] + m,
            zbar,
        )
        G /= pow((1.0 + self._c[0]) * (1.0 - z), 1 + j)
        return G

    def _computeG(self, alpha, beta):
        # TODO: Compute these recursively
        Gfac = self._getGfac(alpha, beta)
        G = np.zeros((self._KMAX + 1, self._JMAX + 1, self._MMAX + 1))
        for k in range(self._KMAX + 1):
            for m in range(self._MMAX + 1):
                for j in range(self._JMAX + 1):
                    G[k, j, m] = self._G_num(alpha, beta, j, k, m)
                G[k, :, m] *= Gfac[k, m]
        return G

    def _computeH(self, alpha, beta):
        G = self._computeG(alpha, beta)
        H = np.zeros((self._KMAX + 1, self._IMAX + 1, self._JMAX + 1))
        for k in range(self._KMAX + 1):
            for i in range(self._IMAX + 1):
                H[k, i] = (G[k] @ self._W[i].reshape(-1, 1)).reshape(-1)
        return H

    def __call__(self, alpha, beta):

        # Initialize the output
        q = np.zeros(self.N)
        Q = np.zeros((self.N, self.N))

        # Hypergeometric sequences
        H = self._computeH(alpha, beta)
        J = lambda i, j: H[0, i, j] + H[1, i, j]
        K = lambda i, j: H[0, i, j] + 2 * H[1, i, j] + H[2, i, j]

        # Case 1: l = 0
        q[0] = -0.5 * J(0, 0)

        # Case 1: l = l' = 0
        Q[0, 0] = 0.25 * K(1, 1)

        # Outer loop
        for l in range(1, self.ydeg + 1):

            n = l * (l + 1)
            sql = 1.0 / np.sqrt(2 * l + 1.0)

            # Case 2: l > 0
            q[n] = -0.5 * sql * (2 * J(0, l) + J(1, l))

            # Case 2: l > 0, l' = 0
            Q[n, 0] = 0.25 * sql * (2 * K(1, l + 1) + K(2, l + 1))
            Q[0, n] = Q[n, 0]

            # Inner loop
            for lp in range(1, l + 1):

                npr = lp * (lp + 1)
                sqlp = 1.0 / np.sqrt(2 * lp + 1.0)

                # Case 3: l > 0, l' > 0
                v = l + lp + 1
                Q[n, npr] = (
                    0.25 * sql * sqlp * (4 * K(1, v) + 4 * K(2, v) + K(3, v))
                )
                Q[npr, n] = Q[n, npr]

        return q, None, None, Q, None, None
