from .integrals import MomentIntegral
from .transforms import ContrastTransform
from .math import cast
from .defaults import defaults
import theano.tensor as tt


class ContrastIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        self.transform = ContrastTransform(**kwargs)

    def _set_params(
        self, mu_c=defaults["mu_c"], sigma_c=defaults["sigma_c"], **kwargs
    ):
        self.mu_c = cast(mu_c)
        self.sigma_c = cast(sigma_c)
        self.fac1 = self.mu_c
        self.fac2 = tt.sqrt(self.sigma_c ** 2 + self.mu_c ** 2)

    def _first_moment(self, e):
        return self.fac1 * e

    def _second_moment(self, eigE):
        return self.fac2 * eigE

    def _log_jac(self):
        return cast(0.0)
