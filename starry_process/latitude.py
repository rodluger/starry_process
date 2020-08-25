from .wigner import R
from .integrals import WignerIntegral
from .ops import LatitudeIntegralOp


class LatitudeIntegral(WignerIntegral):
    def _precompute(self, **kwargs):
        self.R = R(
            self.ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )
        self._integral_op = LatitudeIntegralOp(self.ydeg)

    def _compute_basis_integrals(self, alpha, beta):
        self.q, _, _, self.Q, _, _ = self._integral_op(alpha, beta)
