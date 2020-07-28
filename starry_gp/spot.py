from .transform import eigen, get_alpha_beta
import numpy as np
from scipy.special import hyp2f1

# DEBUG

from scipy.stats import beta as Beta
from scipy.stats import lognorm


def second_integral(ydeg, mu_r=0.1, nu_r=0.01, mu_d=-3.0, nu_d=1.0, c=1.0, K=1000000):
    # Draw
    alpha, beta = get_alpha_beta(mu_r, nu_r)
    r = Beta.rvs(alpha, beta, size=K)
    delta = 1 - lognorm.rvs(s=np.sqrt(nu_d), scale=np.exp(mu_d), size=K)

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

    The spot contrast `delta` is distributed according to:

        1 - delta ~ LogNormal(mu, sig^2)

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

    def set_params(self, mu_r=0.1, nu_r=0.01, mu_d=-3.0, nu_d=1.0):
        alpha, beta = get_alpha_beta(mu_r, nu_r)
        gamma = 1 - np.exp(mu_d + 0.5 * nu_d)

        # Integral normalization
        self.norm1 = gamma * alpha / (alpha + beta)
        self.norm2 = 0.5 * (alpha + 1) / (alpha + beta + 1)

        # Hypergeometric sequence
        self.F = np.zeros(self.ydeg + 1)
        self.G = np.zeros(self.ydeg + 1)
        for l in range(self.ydeg + 1):
            self.F[l] = hyp2f1(l + 1, alpha + 1, alpha + beta + 1, -self.c)
            self.G[l] = hyp2f1(l + 1, alpha + 2, alpha + beta + 2, -self.c)

        # DEBUG
        self.C = second_integral(self.ydeg, mu_r, nu_r, mu_d, nu_d, self.c)

    def first_moment(self):
        """
        Returns the first moment `E[x]` of the spot size 
        and amplitude distribution.

        """
        s = np.zeros(self.ydeg + 1)
        s[0] = 1 - 0.5 * self.norm1 * self.F[0]
        for l in range(1, self.ydeg + 1):
            s[l] = (
                -self.norm1 / np.sqrt(2 * l + 1) * (self.F[l] + self.norm2 * self.G[l])
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
