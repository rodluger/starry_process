from .wigner import R
from .integrals import WignerIntegral
from .transforms import LatitudeTransform
from starry._c_ops import LatitudeGP
import numpy as np


class LatitudeIntegral(WignerIntegral):
    def _precompute(self, **kwargs):
        self.transform = LatitudeTransform(self.ydeg)
        self.R = R(
            self.ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )
        self._c_op = LatitudeGP(self.ydeg)

    def _compute_basis_integrals(self, mean, std):
        alpha, beta = self.transform.get_standard_params(mean, std)
        self._c_op.compute(alpha, beta)
        self.q = self._c_op.q
        self.Q = self._c_op.Q
