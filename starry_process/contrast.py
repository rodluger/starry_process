from .integrals import MomentIntegral
from .transforms import ContrastTransform
from .ops import CheckBoundsOp
from .math import cast
from .defaults import defaults
import theano.tensor as tt
import numpy as np


class ContrastIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        self.transform = ContrastTransform(**kwargs)
        self.check_bounds_ca = CheckBoundsOp(name="ca", lower=-np.inf, upper=1)
        self.check_bounds_cb = CheckBoundsOp(name="cb", lower=0, upper=np.inf)

    def _set_params(self, ca=defaults["ca"], cb=defaults["cb"], **kwargs):
        self.mu_c = self.check_bounds_ca(ca)
        self.sigma_c = self.check_bounds_cb(cb)
        self.fac1 = self.mu_c
        self.fac2 = tt.sqrt(self.sigma_c ** 2 + self.mu_c ** 2)

    def _first_moment(self, e):
        return self.fac1 * e

    def _second_moment(self, eigE):
        return self.fac2 * eigE

    def _log_jac(self):
        return cast(0.0)
