from .transforms import SizeTransform
from .integrals import MomentIntegral
from .utils import eigen
import numpy as np
from scipy.special import hyp2f1
from scipy.special import beta as EulerBeta


# Dimensions & misc. constants
imax = 3
kmax = 2
mmax = 4
V = np.array(
    [[1, 1, 0, 0, 0], [1, 2, 1, 0, 0], [1, 3, 3, 1, 0], [1, 4, 6, 4, 1]],
    dtype=int,
)


def _get_G_factor(alpha, beta, c):
    kmax = 2
    mmax = 4
    lam = np.zeros((kmax + 1, mmax + 1))
    k = np.arange(kmax + 1)
    q = c[1] / c[0]
    lam[:, 0] = 1.0
    for m in range(1, mmax + 1):
        for k in range(kmax + 1):
            lam[k, m] = (
                lam[k, m - 1]
                * q
                * (alpha + m - 1)
                / (alpha + beta + k * c[3] + m - 1)
            )
    norm = 1 / EulerBeta(alpha, beta)
    for k in range(1, kmax + 1):
        lam[k] *= c[2] ** k * (norm * EulerBeta(alpha, beta + k * c[3]))
    return lam


def get_G(ydeg, alpha, beta, c, tol=1e-6):
    """
    Return the hypergeometric G_{jkm} tensor, computed via tridiagonal recursion.

    """
    # Dimensions
    jmax = 2 * ydeg + 1
    G = np.zeros((jmax + 1, kmax + 1, mmax + 1))

    # Shorthand
    q = c[1] / c[0]
    p = 1 + c[0] + c[1]
    z = -c[1] / (1.0 + c[0])
    zbar = c[1] / p
    betap = lambda k: beta + k * c[3]

    # The exact function, computed numerically
    G_num = (
        lambda j, k, m: hyp2f1(1 + j, betap(k), alpha + betap(k) + m, zbar)
        / (1 + c[0]) ** (j + 1)
        * pow(1 - z, -(j + 1))
    )

    # The recursion coefficients
    a = lambda j, k, m: -((alpha + betap(k) + m - j) * (1 + c[0])) / (
        j * p * (1 + c[0]) ** 2
    )
    b = lambda j, k, m: -(
        1
        - ((alpha + betap(k) + m - j) * (1 + c[0]) + (alpha + m) * c[1])
        / (j * p)
    ) / (1 + c[0])

    # Solve the problem for each value of `k`
    for k in range(kmax + 1):

        # Boundary conditions
        G[0, k, 0] = G_num(0, k, 0)
        G[0, k, 1] = G_num(0, k, 1)
        G[jmax, k, 0] = G_num(jmax, k, 0)
        G[jmax, k, 1] = G_num(jmax, k, 1)

        # Recurse upward in m
        for j in [0, jmax]:
            for m in range(2, mmax + 1):
                # Be careful about division by zero here
                if np.abs(alpha + betap(k) - j + m - 2) < tol:
                    G[j, k, m] = G_num(j, k, m)
                else:
                    term = (alpha + m - 1) / (alpha + betap(k) + m - 1)
                    am = ((alpha + betap(k) + m - 2) * (1 + c[0])) / (
                        (alpha + betap(k) - j + m - 2) * term * c[1]
                    )
                    bm = 1.0 / term - (
                        (alpha + betap(k) + m - 2) * (1 + c[0])
                        + betap(k) * c[1]
                    ) / ((alpha + betap(k) - j + m - 2) * term * c[1])
                    G[j, k, m] = am * G[j, k, m - 2] + bm * G[j, k, m - 1]

        # Recurse along the j dimension
        for m in range(mmax + 1):
            # The tridiagonal matrix system, M G = x
            M = (
                np.diag([1 for j in range(2, jmax)], k=1)
                + np.diag([b(j, k, m) for j in range(2, jmax + 1)])
                + np.diag([a(j, k, m) for j in range(3, jmax + 1)], k=-1)
            )
            x = np.zeros(jmax - 1)
            x[0] = -a(2, k, m) * G[0, k, m]
            x[-1] = -G[jmax, k, m]

            # Solve it
            G[1:-1, k, m] = np.linalg.solve(M, x)

    # Finally, apply the amplitude factor
    G *= _get_G_factor(alpha, beta, c).reshape(1, kmax + 1, mmax + 1)

    return G


def get_H(ydeg, alpha, beta, c):
    jmax = 2 * ydeg + 1
    G = get_G(ydeg, alpha, beta, c)
    H = np.zeros((imax + 1, jmax + 1, kmax + 1))
    fac = c[0] ** np.arange(1, imax + 2).reshape(-1, 1)
    W = fac * V
    for i in range(imax + 1):
        H[i] = G @ W[i]
    return H


class SizeIntegral(MomentIntegral):
    """Marginalizes over the spot size distribution.

    Computes the first two moments of the distribution over spherical
    harmonic coefficients given a distribution of spot sizes.

    """

    def _precompute(self, **kwargs):
        self.transform = SizeTransform(self.ydeg, **kwargs)
        l = np.arange(self.ydeg + 1)
        self.i = l * (l + 1)
        self.ij = np.ix_(self.i, self.i)
        self._e = np.zeros(self.N)
        self._eigE = np.zeros((self.N, self.N))

    def _set_params(self, mean, std):

        # Get the Beta params
        alpha, beta = self.transform.get_standard_params(mean, std)

        # Hypergeometric sequences
        H = get_H(self.ydeg, alpha, beta, self.transform.c)
        J = lambda i, j: H[i, j, 0] + H[i, j, 1]
        K = lambda i, j: H[i, j, 0] + 2 * H[i, j, 1] + H[i, j, 2]

        # Compact (m = 0) moment matrices
        e = np.zeros(self.ydeg + 1)
        E = np.zeros((self.ydeg + 1, self.ydeg + 1))

        # Case 1: l = 0
        e[0] = -0.5 * J(0, 0)

        # Case 1: l = l' = 0
        E[0, 0] = 0.25 * K(1, 1)

        # Outer loop
        for l in range(1, self.ydeg + 1):

            sql = 1.0 / np.sqrt(2 * l + 1)

            # Case 2: l > 0
            e[l] = -0.5 * sql * (2 * J(0, l) + J(1, l))

            # Case 2: l > 0, l' = 0
            E[l, 0] = 0.25 * sql * (2 * K(1, l + 1) + K(2, l + 1))
            E[0, l] = E[l, 0]

            # Inner loop
            for lp in range(1, l + 1):

                sqlp = 1.0 / np.sqrt(2 * lp + 1)

                # Case 3: l > 0, l' > 0
                n = l + lp + 1
                E[l, lp] = (
                    0.25 * sql * sqlp * (4 * K(1, n) + 4 * K(2, n) + K(3, n))
                )
                E[lp, l] = E[l, lp]

        # Assemble the full (sparse) moment matrices
        self._e[self.i] = e
        self._eigE[self.ij] = eigen(E)

    def _first_moment(self):
        return self._e

    def _second_moment(self):
        return self._eigE
