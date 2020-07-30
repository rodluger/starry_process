from .transform import eigen, get_alpha_beta
import numpy as np
from scipy.special import hyp2f1


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
        self.E1 = np.zeros(self.N)
        self.E2 = np.zeros((self.N, self.N))
        self.set_params()

    def set_params(self, mu_r=0.1, nu_r=0.01, mu_b=-3.0, nu_b=1.0):

        # Get the Beta params
        alpha, beta = get_alpha_beta(mu_r, nu_r)

        # Shorthand
        g1 = 1 - np.exp(mu_b + 0.5 * nu_b)
        g2 = 1 - 2 * np.exp(mu_b + 0.5 * nu_b) + np.exp(2 * mu_b + 2 * nu_b)
        ab = alpha + beta
        c = self.c
        c0 = self.c / (self.c + 1)
        p0 = alpha / ab
        p1 = (alpha + 1) / (ab + 1)
        p2 = (alpha + 2) / (ab + 2)
        p3 = (alpha + 3) / (ab + 3)
        gr = g2 / g1

        # Hypergeometric sequence
        G = np.zeros((2 * self.ydeg + 2, 4))

        # Compute the first few terms explicitly
        for j in range(2):
            for k in range(4):
                G[j, k] = hyp2f1(j + 1, alpha + k + 1, ab + k + 1, -c)

        # Now recurse upward
        for j in range(2, 2 * self.ydeg + 2):
            for k in range(4):
                G[j, k] = (
                    (ab + k + 1 - j) * G[j - 2, k]
                    - (2 * alpha + beta + 2 * k + 2 - 3 * j) * G[j - 1, k]
                ) / (2.0 * j)

        # Moment matrices
        s = np.zeros(self.ydeg + 1)
        S = np.zeros((self.ydeg + 1, self.ydeg + 1))

        # Case 1: l = 0
        s[0] = 1 - 0.5 * g1 * p0 * G[0, 0]

        # Case 1: l = l' = 0
        S[0, 0] = (1 + 0.25 * (p0 * alpha + p0) * c * c0 * g2) - (p0 * g1 * c) * (
            G[0, 0] + (0.25 * p1 * gr * c0 * (ab + c * (alpha + 1))) * G[0, 1]
        )

        # Outer loop
        for l in range(1, self.ydeg + 1):
            sql = 1 / np.sqrt(2 * l + 1)

            # Case 2: l > 0
            s[l] = -g1 * p0 * sql * (G[l, 0] + 0.5 * p1 * G[l, 1])

            # Case 2: l > 0, l' = 0
            S[l, 0] = (-0.5 * p0 * g1 * c * sql) * (
                2 * G[l, 0]
                + c * p1 * G[l, 1]
                - c * gr * p1 * G[l + 1, 1]
                - 0.5 * c ** 2 * gr * p1 * p2 * G[l + 1, 2]
            )
            S[0, l] = S[l, 0]

            # Inner loop
            for lp in range(1, l + 1):
                sqp = 1 / np.sqrt(2 * lp + 1)

                # Case 3: l > 0, l' > 0
                S[l, lp] = (p0 * p1 * g2 * c ** 2 * sql * sqp) * (
                    G[l + lp + 1, 1]
                    + c * p2 * G[l + lp + 1, 2]
                    + 0.25 * c ** 2 * p2 * p3 * G[l + lp + 1, 3]
                )
                S[lp, l] = S[l, lp]

        # Assemble the full matrices
        self.E1[self.i] = s
        self.E2[self.ij] = eigen(S)

    def first_moment(self):
        """
        Returns the first moment `E[y_lm]` of the spot size 
        and amplitude distribution.

        """
        return self.E1

    def second_moment(self):
        """
        Returns the eigendecomposition `C` of the second moment 
        `E[y_lm y_l'm']` of the spot size and amplitude distribution, 
        such that

            C @ C.T = E[y_lm y_l'm']

        """
        return self.E2
