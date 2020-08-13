from .transform import TransformIntegral, get_alpha_beta
from starry._c_ops import LatitudeGP
import numpy as np


class LatitudeIntegral(TransformIntegral):
    def __init__(self, ydeg):
        self._c_op = LatitudeGP(ydeg)
        super().__init__(
            ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )
        self.set_params()

    def _compute_basis_integrals(self, mu=0.5, nu=0.01):
        alpha, beta = get_alpha_beta(mu, nu)
        self._c_op.compute(alpha, beta)
        self.q = self._c_op.q
        self.Q = self._c_op.Q
