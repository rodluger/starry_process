from .integrals import MomentIntegral
from .transforms import SizeTransform


class SizeIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        self.transform = SizeTransform(ydeg=self.ydeg, **kwargs)
        kwargs.update(
            {
                "SP__C0": "{:.15f}".format(self.transform._c[0]),
                "SP__C1": "{:.15f}".format(self.transform._c[1]),
                "SP__C2": "{:.15f}".format(self.transform._c[2]),
                "SP__C3": "{:.15f}".format(self.transform._c[3]),
            }
        )
        self._integral_op = self._math.ops.SizeIntegralOp(self.ydeg, **kwargs)

    def _set_params(self, alpha, beta):
        alpha = self._math.cast(alpha)
        beta = self._math.cast(beta)
        self.q, _, _, self.Q, _, _ = self._integral_op(alpha, beta)
        self.eigQ = self._math.eigen(self.Q)

    def _first_moment(self):
        return self.q

    def _second_moment(self):
        return self.eigQ
