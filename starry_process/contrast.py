from .integrals import MomentIntegral
from .transforms import ContrastTransform


class ContrastIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        self.transform = ContrastTransform(**kwargs)

    def _set_params(self, mu, sigma):
        mu = self._math.cast(mu)
        sigma = self._math.cast(sigma)
        self.fac1 = mu
        self.fac2 = self._math.sqrt(sigma ** 2 + mu ** 2)

    def _first_moment(self, e):
        return self.fac1 * e

    def _second_moment(self, eigE):
        return self.fac2 * eigE
