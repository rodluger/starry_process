from .transform import eigen, get_alpha_beta
import numpy as np
from scipy.special import hyp2f1


class SpotIntegral(object):
    """Marginalizes over the spot size and amplitude distribution.

    Computes the first two moments of the distribution over spherical
    harmonic coefficients given a distribution of spot sizes and amplitudes.

    The spot size `s` is Beta-distributed:

        s ~ Beta(alpha, beta)
    
    where `alpha` and `beta` are parametrized in terms of `mu` and `nu`:

        alpha = mu * (1 / nu - 1)
        beta = (1 - mu) * (1 / nu - 1)

    where `mu` and `nu` are the mean and (normalized) variance of the
    distribution.

    The spot amplitude `a` is log-normally distributed:

        a ~ LogNormal(mu, nu)

    """

    def __init__(self, ydeg):
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2
        l = np.arange(1, ydeg + 1)
        self.i = l * (l + 1)
        self.ij = np.ix_(self.i, self.i)
        self.set_params()

    def set_params(self, mu_s=0.1, nu_s=0.01, mu_a=-3.0, nu_a=1.0):
        self.alpha_s, self.beta_s = get_alpha_beta(mu_s, nu_s)
        self._ampfac1 = -np.exp(mu_a + 0.5 * nu_a)
        self._ampfac2 = np.exp(2 * mu_a + 2 * nu_a)

    def first_moment(self):
        """
        Returns the first moment `E[x]` of the spot size 
        and amplitude distribution.

        """
        s = np.zeros(self.ydeg)
        for k in range(self.ydeg):
            l = k + 1
            s[k] = hyp2f1(self.alpha_s, l ** 2, self.alpha_s + self.beta_s, -1)
        S = np.zeros(self.N)
        S[self.i] = s * self._ampfac1
        return S

    def second_moment(self):
        """
        Returns the eigendecomposition `C` of the second moment `E[x^2]` 
        of the spot size and amplitude distribution, such that

            C @ C.T = E[x^2]

        """
        c = np.zeros((self.ydeg, self.ydeg))
        for k1 in range(self.ydeg):
            l1 = k1 + 1
            for k2 in range(self.ydeg):
                l2 = k2 + 1
                c[k1, k2] = hyp2f1(
                    self.alpha_s, l1 ** 2 + l2 ** 2, self.alpha_s + self.beta_s, -1
                )
        C = np.zeros((self.N, self.N))
        C[self.ij] = c * self._ampfac2
        return eigen(C)
