from .integrals import MomentIntegral
import numpy as np


class ContrastIntegral(MomentIntegral):
    """Marginalizes over the spot contrast distribution.

    Computes the first two moments of the distribution over spherical
    harmonic coefficients given a distribution of spot contrasts.

    The spot contrast `xi` is distributed according to:

        b ~ LogNormal(mu, nu)
        xi = 1 - b
    
    where `mu` and `nu` are the mean and variance of the
    brightness distribution.

    """

    def _precompute(self, **kwargs):
        pass

    def _set_params(self, mu, nu):
        assert nu > 0, "variance must be positive."
        self.fac1 = 1 - np.exp(mu + 0.5 * nu)
        self.fac2 = np.sqrt(
            1 - 2 * np.exp(mu + 0.5 * nu) + np.exp(2 * mu + 2 * nu)
        )

    def _first_moment(self, e):
        return self.fac1 * e

    def _second_moment(self, eigE):
        return self.fac2 * eigE
