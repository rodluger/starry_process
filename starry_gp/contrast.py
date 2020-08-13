import numpy as np


class ContrastIntegral(object):
    """Marginalizes over the spot contrast distribution.

    Computes the first two moments of the distribution over spherical
    harmonic coefficients given a distribution of spot contrasts.

    The spot contrast `xi` is distributed according to:

        b ~ LogNormal(mu, nu)
        xi = 1 - b
    
    where `mu` and `nu` are the mean and variance of the
    brightness distribution.

    """

    def __init__(self, ydeg):
        self.ydeg = ydeg
        self.fac1 = 0
        self.fac2 = 0
        self.set_params()

    def set_params(self, mu=-0.1, nu=0.01):
        self.fac1 = 1 - np.exp(mu + 0.5 * nu)
        self.fac2 = np.sqrt(
            1 - 2 * np.exp(mu + 0.5 * nu) + np.exp(2 * mu + 2 * nu)
        )

    def first_moment(self, e):
        """
        Returns the first moment of the spot contrast 
        distribution.

        """
        return self.fac1 * e

    def second_moment(self, eigE):
        """
        Returns the eigendecomposition of the second moment 
        of the spot contrast distribution.

        """
        return self.fac2 * eigE
