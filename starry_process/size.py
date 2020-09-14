from .integrals import MomentIntegral
from .transforms import SizeTransform
from .math import cast, matrix_sqrt
from .ops import SizeIntegralOp, CheckBoundsOp
from .defaults import defaults
import theano.tensor as tt


class SizeIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        self.transform = SizeTransform(ydeg=self.ydeg, **kwargs)
        kwargs.update(
            {
                "compile_args": kwargs.get("compile_args", [])
                + [
                    ("SP__C0", "{:.15f}".format(self.transform._c[0])),
                    ("SP__C1", "{:.15f}".format(self.transform._c[1])),
                    ("SP__C2", "{:.15f}".format(self.transform._c[2])),
                    ("SP__C3", "{:.15f}".format(self.transform._c[3])),
                ]
            }
        )
        self._integral_op = SizeIntegralOp(self.ydeg, **kwargs)
        self.check_bounds_sa = CheckBoundsOp(name="sa", lower=0, upper=1)
        self.check_bounds_sb = CheckBoundsOp(name="sb", lower=0, upper=1)

    def _set_params(self, sa=defaults["sa"], sb=defaults["sb"], **kwargs):
        self.sa = self.check_bounds_sa(sa)
        sa1 = self.transform._ln_alpha_min
        sa2 = self.transform._ln_alpha_max
        alpha_s = tt.exp(cast(sa1 + self.sa * (sa2 - sa1)))
        self.sb = self.check_bounds_sb(sb)
        sb1 = self.transform._ln_beta_min
        sb2 = self.transform._ln_beta_max
        beta_s = tt.exp(cast(sb1 + self.sb * (sb2 - sb1)))
        self.q, _, _, self.Q, _, _ = self._integral_op(alpha_s, beta_s)
        self.eigQ = matrix_sqrt(self.Q, driver=self.driver)

    def _first_moment(self, e=None):
        return self.q

    def _second_moment(self, eigE=None):
        return self.eigQ

    def _log_jac(self):
        return self.transform.log_jac(self.sa, self.sb)
