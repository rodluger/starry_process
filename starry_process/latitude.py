from .wigner import R
from .integrals import WignerIntegral
from .ops import LatitudeIntegralOp
from .transforms import LatitudeTransform
import theano.tensor as tt


class LatitudeIntegral(WignerIntegral):
    def _precompute(self, **kwargs):
        self.transform = LatitudeTransform(**kwargs)
        self.R = R(
            self.ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )
        self._integral_op = LatitudeIntegralOp(self.ydeg)

    def _compute_basis_integrals(self, alpha, beta):
        alpha = tt.as_tensor_variable(alpha).astype(tt.config.floatX)
        beta = tt.as_tensor_variable(beta).astype(tt.config.floatX)
        self.q, _, _, self.Q, _, _ = self._integral_op(alpha, beta)
