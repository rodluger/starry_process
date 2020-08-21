from .integrals import MomentIntegral
from .transforms import ContrastTransform
import numpy as np


class ContrastIntegral(MomentIntegral):
    """Marginalizes over the spot contrast distribution.

    Computes the first two moments of the distribution over spherical
    harmonic coefficients given a distribution of spot contrasts.

    The spot contrast `xi` is distributed according to:

        b ~ LogNormal(mu, sigma^2)
        xi = 1 - b
    
    where `mu` and `sigma` are the mean and std. dev. of the
    brightness distribution.

    """

    def _precompute(self, **kwargs):
        self.transform = ContrastTransform(self.ydeg)

    def _set_params(self, mean, std):
        assert std > 0, "std is out of bounds"
        var = std ** 2
        self.fac1 = 1 - np.exp(mean + 0.5 * var)
        self.fac2 = np.sqrt(
            1 - 2 * np.exp(mean + 0.5 * var) + np.exp(2 * mean + 2 * var)
        )

    def _first_moment(self, e):
        return self.fac1 * e

    def _second_moment(self, eigE):
        return self.fac2 * eigE
