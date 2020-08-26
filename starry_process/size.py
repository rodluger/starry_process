from .integrals import MomentIntegral, eigen
from .ops import SizeIntegralOp


class SizeIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        self._integral_op = SizeIntegralOp(self.ydeg)

    def _set_params(self, alpha, beta):
        self.q, _, _, self.Q, _, _ = self._integral_op(alpha, beta)
        self.eigQ = eigen(self.Q)

    def _first_moment(self):
        return self.q

    def _second_moment(self):
        return self.eigQ
