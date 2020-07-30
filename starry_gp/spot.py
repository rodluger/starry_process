from .transform import eigen, get_alpha_beta
import numpy as np
from scipy.special import hyp2f1

# DEBUG

from scipy.stats import beta as Beta
from scipy.stats import lognorm


def second_integral(ydeg, mu_r=0.1, nu_r=0.01, mu_b=-3.0, nu_b=1.0, c=1.0, K=1000000):
    # Draw
    alpha, beta = get_alpha_beta(mu_r, nu_r)
    r = Beta.rvs(alpha, beta, size=K)
    delta = 1 - lognorm.rvs(s=np.sqrt(nu_b), scale=np.exp(mu_b), size=K)

    # Transform
    s = np.zeros((K, ydeg + 1))
    s[:, 0] = 1 - 0.5 * delta * c * r / (1 + c * r)
    for l in range(1, ydeg + 1):
        s[:, l] = (
            -0.5
            * delta
            * c
            * r
            * (2 + c * r)
            / (np.sqrt(2 * l + 1) * (1 + c * r) ** (l + 1))
        )
    l = np.arange(ydeg + 1)
    S = np.zeros((K, (ydeg + 1) * (ydeg + 1)))
    S[:, l * (l + 1)] = s

    # Compute second moment
    mu = np.mean(S, axis=0)
    cov = np.cov(S.T)
    return cov + mu.reshape(-1, 1) @ mu.reshape(1, -1)


class SpotIntegral(object):
    """Marginalizes over the spot size and contrast distribution.

    Computes the first two moments of the distribution over spherical
    harmonic coefficients given a distribution of spot sizes and contrasts.

    The spot size `r` is Beta-distributed:

        r ~ Beta(alpha, beta)
    
    where `alpha` and `beta` are parametrized in terms of `mu` and `nu`:

        alpha = mu * (1 / nu - 1)
        beta = (1 - mu) * (1 / nu - 1)

    where `mu` and `nu` are the mean and (normalized) variance of the
    distribution.

    The spot brightness `b` is distributed according to:

        b ~ LogNormal(mu, sig^2)

    where the parameters are the mean `mu` and the variance `nu = sig^2`
    of the Log Normal.

    """

    def __init__(self, ydeg, c=1.0):
        self.ydeg = ydeg
        self.c = c
        self.N = (ydeg + 1) ** 2
        l = np.arange(ydeg + 1)
        self.i = l * (l + 1)
        self.ij = np.ix_(self.i, self.i)
        self.set_params()

    def set_params(self, mu_r=0.1, nu_r=0.01, mu_b=-3.0, nu_b=1.0):
        alpha, beta = get_alpha_beta(mu_r, nu_r)
        g1 = 1 - np.exp(mu_b + 0.5 * nu_b)
        g2 = 1 - 2 * np.exp(mu_b + 0.5 * nu_b) + np.exp(2 * mu_b + 2 * nu_b)

        # Integral normalization
        self.norm1 = g1 * alpha / (alpha + beta)
        self.norm2 = 0.5 * (alpha + 1) / (alpha + beta + 1)

        # Hypergeometric sequence
        self.G = np.zeros((2 * self.ydeg + 2, 4))
        for l in range(2 * self.ydeg + 2):
            # TODO: Recurse
            self.G[l, 0] = hyp2f1(l + 1, alpha + 1, alpha + beta + 1, -self.c)
            self.G[l, 1] = hyp2f1(l + 1, alpha + 2, alpha + beta + 2, -self.c)
            self.G[l, 2] = hyp2f1(l + 1, alpha + 3, alpha + beta + 3, -self.c)
            self.G[l, 3] = hyp2f1(l + 1, alpha + 4, alpha + beta + 4, -self.c)

        # TODO: Move this
        s = np.zeros((self.ydeg + 1, self.ydeg + 1))

        # Shorthand
        ab = alpha + beta
        c = self.c
        c0 = self.c / (self.c + 1)
        p0 = alpha / ab
        p1 = (alpha + 1) / (ab + 1)
        p2 = (alpha + 2) / (ab + 2)
        p3 = (alpha + 3) / (ab + 3)
        gr = g2 / g1

        # Case 1: l = l' = 0
        t1 = 1 + 0.25 * (p0 * alpha + p0) * c * c0 * g2
        t2 = 0.25 * p1 * gr * c0 * (ab + c * (alpha + 1))
        s[0, 0] = t1 - p0 * g1 * c * (self.G[0, 0] + t2 * self.G[0, 1])

        # Case 2: l > 0, l' = 0
        for l in range(1, self.ydeg + 1):
            s[l, 0] = (-0.5 * p0 * g1 * c / np.sqrt(2 * l + 1)) * (
                2 * self.G[l, 0]
                + c * p1 * self.G[l, 1]
                - c * gr * p1 * self.G[l + 1, 1]
                - 0.5 * c ** 2 * gr * p1 * p2 * self.G[l + 1, 2]
            )
            s[0, l] = s[l, 0]

        # Case 3: l > 0, l' > 0
        for l in range(1, self.ydeg + 1):
            sql = 1 / np.sqrt(2 * l + 1)
            for lp in range(1, l + 1):
                sqp = 1 / np.sqrt(2 * lp + 1)
                s[l, lp] = (p0 * p1 * g2 * c ** 2 * sql * sqp) * (
                    self.G[l + lp + 1, 1]
                    + c * p2 * self.G[l + lp + 1, 2]
                    + 0.25 * c ** 2 * p2 * p3 * self.G[l + lp + 1, 3]
                )
                s[lp, l] = s[l, lp]

        # Construct the sparse matrix
        self.C = np.zeros((self.N, self.N))
        self.C[self.ij] = s

    def first_moment(self):
        """
        Returns the first moment `E[x]` of the spot size 
        and amplitude distribution.

        """
        s = np.zeros(self.ydeg + 1)
        s[0] = 1 - 0.5 * self.norm1 * self.G[0, 0]
        for l in range(1, self.ydeg + 1):
            s[l] = (
                -self.norm1
                / np.sqrt(2 * l + 1)
                * (self.G[l, 0] + self.norm2 * self.G[l, 1])
            )
        S = np.zeros(self.N)
        S[self.i] = s
        return S

    def second_moment(self):
        """
        Returns the eigendecomposition `C` of the second moment `E[x^2]` 
        of the spot size and amplitude distribution, such that

            C @ C.T = E[x^2]

        """
        # DEBUG
        return eigen(self.C)
