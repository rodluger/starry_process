from .integrals import MomentIntegral
from .transforms import eigen, get_c0_c1, get_alpha_beta
import numpy as np
from scipy.special import hyp2f1


def get_s(ydeg, r, tol=1e-2, hwhm_max=75):
    """Return the spot spherical harmonic expansion vector `s(r)`.

    """
    c0, c1 = get_c0_c1(ydeg, tol=tol, hwhm_max=hwhm_max)
    r = np.atleast_1d(r)
    assert len(r.shape) == 1
    K = r.shape[0]
    rprime = c0 + c1 * r
    sm0 = np.zeros((K, ydeg + 1))
    sm0[:, 0] = 0.5 * rprime
    for l in range(ydeg + 1):
        sm0[:, l] -= (
            rprime
            * (2 + rprime)
            / (2 * np.sqrt(2 * l + 1) * (1 + rprime) ** (l + 1))
        )
    l = np.arange(ydeg + 1)
    s = np.zeros((K, (ydeg + 1) * (ydeg + 1)))
    s[:, l * (l + 1)] = sm0
    return s


def get_G(alpha, beta, c0, c1, jmax, tol=1e-6):
    """
    Return the hypergeometric G_j^k matrix,
    computed via tridiagonal recursion.

    """
    # Shorthand
    z = -c1 / (1.0 + c0)
    ab = alpha + beta
    lam = np.array(
        [
            alpha / ab,
            (alpha + 1) / (ab + 1),
            (alpha + 2) / (ab + 2),
            (alpha + 3) / (ab + 3),
        ]
    )

    # The exact function, computed numerically
    G_num = (
        lambda j, k: hyp2f1(1 + j, alpha + k, ab + k, z)
        if np.abs(z) < 1
        else hyp2f1(1 + j, beta, ab + k, z / (z - 1)) * pow(1 - z, -j - 1)
    )

    # The recursion coefficients
    a = lambda j, k: -((alpha + beta + k - j) * (1 + c0)) / (j * (1 + c0 + c1))
    b = lambda j, k: -(
        1
        - ((alpha + beta + k - j) * (1 + c0) + (alpha + k) * c1)
        / (j * (1 + c0 + c1))
    )

    # Boundary conditions
    G = np.zeros((jmax + 1, 5))
    G[0, 0] = G_num(0, 0)
    G[0, 1] = G_num(0, 1)
    G[jmax, 0] = G_num(jmax, 0)
    G[jmax, 1] = G_num(jmax, 1)

    # Recurse upward in k
    for j in [0, jmax]:
        for k in range(2, 5):
            # Be careful about division by zero here
            if np.abs(alpha + beta - j + k - 2) < tol:
                G[j, k] = G_num(j, k)
            else:
                ak = ((alpha + beta + k - 2) * (1 + c0)) / (
                    (alpha + beta - j + k - 2) * lam[k - 1] * c1
                )
                bk = 1.0 / lam[k - 1] - (
                    (alpha + beta + k - 2) * (1 + c0) + beta * c1
                ) / ((alpha + beta - j + k - 2) * lam[k - 1] * c1)
                G[j, k] = ak * G[j, k - 2] + bk * G[j, k - 1]

    # Recurse along the j dimension
    for k in range(5):
        # The tridiagonal matrix system, M G = c
        M = (
            np.diag([1 for j in range(2, jmax)], k=1)
            + np.diag([b(j, k) for j in range(2, jmax + 1)])
            + np.diag([a(j, k) for j in range(3, jmax + 1)], k=-1)
        )
        c = np.zeros(jmax - 1)
        c[0] = -a(2, k) * G[0, k]
        c[-1] = -G[jmax, k]

        # Solve it
        G[1:-1, k] = np.linalg.solve(M, c)

    return G


class SizeIntegral(MomentIntegral):
    """Marginalizes over the spot size distribution.

    Computes the first two moments of the distribution over spherical
    harmonic coefficients given a distribution of spot sizes.

    The spot size `r` is Beta-distributed:

        r ~ Beta(alpha, beta)
    
    where `alpha` and `beta` are parametrized in terms of `mu` and `nu`:

        alpha = mu * (1 / nu - 1)
        beta = (1 - mu) * (1 / nu - 1)

    where `mu` and `nu` are the mean and (normalized) variance of the
    distribution.

    """

    def _precompute(self, tol=1e-2, hwhm_max=75, **kwargs):
        self.c0, self.c1 = get_c0_c1(self.ydeg, tol=tol, hwhm_max=hwhm_max)
        l = np.arange(self.ydeg + 1)
        self.i = l * (l + 1)
        self.ij = np.ix_(self.i, self.i)
        self._e = np.zeros(self.N)
        self._eigE = np.zeros((self.N, self.N))

    def _set_params(self, mu, nu):

        # Get the Beta params
        alpha, beta = get_alpha_beta(mu, nu)

        # Shorthand
        c0 = self.c0
        c1 = self.c1
        ab = alpha + beta
        lam = np.array(
            [
                alpha / ab,
                (alpha + 1) / (ab + 1),
                (alpha + 2) / (ab + 2),
                (alpha + 3) / (ab + 3),
            ]
        )

        # Hypergeometric sequence
        G = get_G(alpha, beta, c0, c1, 2 * self.ydeg + 1)
        H = lambda j, k: np.prod(lam[:k]) * G[j, k]

        # Compact (m = 0) moment matrices
        e = np.zeros(self.ydeg + 1)
        E = np.zeros((self.ydeg + 1, self.ydeg + 1))

        # Case 1: l = 0
        e[0] = -1.0 / (2.0 * (1 + c0)) * (c0 * H(0, 0) + c1 * H(0, 1))

        # Case 1: l = l' = 0
        E[0, 0] = (
            1.0
            / (4.0 * (1 + c0) ** 2)
            * (c0 ** 2 * H(1, 0) + 2 * c0 * c1 * H(1, 1) + c1 ** 2 * H(1, 2))
        )

        # Outer loop
        for l in range(1, self.ydeg + 1):
            term = 1.0 / (2 * np.sqrt(2 * l + 1) * (1 + c0) ** (l + 1))

            # Case 2: l > 0
            e[l] = -term * (
                c0 * (2 + c0) * H(l, 0)
                + 2 * c1 * (1 + c0) * H(l, 1)
                + c1 ** 2 * H(l, 2)
            )

            # Case 2: l > 0, l' = 0
            E[l, 0] = (
                term
                / (2.0 * (1 + c0))
                * (
                    c0 ** 2 * (2 + c0) * H(l + 1, 0)
                    + c0 * c1 * (4 + 3 * c0) * H(l + 1, 1)
                    + (2 + 3 * c0) * c1 ** 2 * H(l + 1, 2)
                    + c1 ** 3 * H(l + 1, 3)
                )
            )
            E[0, l] = E[l, 0]

            # Inner loop
            for lp in range(1, l + 1):
                sqp = 1 / np.sqrt(2 * lp + 1)

                # Case 3: l > 0, l' > 0
                E[l, lp] = (
                    term
                    / (2.0 * np.sqrt(2 * lp + 1) * (1 + c0) ** (lp + 1))
                    * (
                        c0 ** 2 * (2 + c0) ** 2 * H(l + lp + 1, 0)
                        + 4 * c0 * (1 + c0) * (2 + c0) * c1 * H(l + lp + 1, 1)
                        + 2
                        * (2 + 3 * c0 * (2 + c0))
                        * c1 ** 2
                        * H(l + lp + 1, 2)
                        + 4 * (1 + c0) * c1 ** 3 * H(l + lp + 1, 3)
                        + c1 ** 4 * H(l + lp + 1, 4)
                    )
                )
                E[lp, l] = E[l, lp]

        # Assemble the full (sparse) moment matrices
        self._e[self.i] = e
        self._eigE[self.ij] = eigen(E)

    def _first_moment(self):
        return self._e

    def _second_moment(self):
        return self._eigE
