from .wigner import R
from .integrals import WignerIntegral
from .transforms import get_alpha_beta
from starry._c_ops import LatitudeGP
import numpy as np


class LatitudeIntegral(WignerIntegral):
    def _precompute(self, **kwargs):
        self.R = R(
            self.ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )
        self._c_op = LatitudeGP(self.ydeg)

    def _compute_basis_integrals(self, mu, nu):
        alpha, beta = get_alpha_beta(mu, nu)
        self._c_op.compute(alpha, beta)
        self.q = self._c_op.q
        self.Q = self._c_op.Q
