from .transform import TransformIntegral
from starry._c_ops import LatitudeGP
import numpy as np
from scipy.special import gamma, hyp2f1


class LatitudeIntegral(TransformIntegral):
    def __init__(self, ydeg):
        self._c_op = LatitudeGP(ydeg)
        super().__init__(ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1)
        self.set_params()

    def _compute_basis_integrals(self, alpha=2.0, beta=2.0):
        self._c_op.compute(alpha, beta)
        self.q = self._c_op.q
        self.Q = self._c_op.Q
