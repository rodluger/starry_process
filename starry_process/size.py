from .integrals import MomentIntegral
from .math import eigen
from .ops import SizeIntegralOp
from .transforms import SizeTransform
import theano.tensor as tt


class SizeIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        self.transform = SizeTransform(ydeg=self.ydeg, **kwargs)
        self._integral_op = SizeIntegralOp(self.ydeg, c=self.transform._c)

    def _set_params(self, alpha, beta):
        alpha = tt.as_tensor_variable(alpha).astype(tt.config.floatX)
        beta = tt.as_tensor_variable(beta).astype(tt.config.floatX)
        self.q, _, _, self.Q, _, _ = self._integral_op(alpha, beta)
        self.eigQ = eigen(self.Q)

    def _first_moment(self):
        return self.q

    def _second_moment(self):
        return self.eigQ
