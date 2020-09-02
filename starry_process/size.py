from .integrals import MomentIntegral
from .transforms import SizeTransform
from .math import cast, eigen
from .ops import SizeIntegralOp
from .defaults import defaults


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
        self._integral_op = SizeIntegralOp(self.ydeg, **kwargs)

    def set_params(
        self, alpha_s=None, beta_s=None, mu_s=None, sigma_s=None, **kwargs
    ):
        p1 = [alpha_s, beta_s]
        p2 = [mu_s, sigma_s]
        if all([p is None for p in p1 + p2]):
            # No params provided: use defaults
            mu_s = defaults["mu_s"]
            sigma_s = defaults["sigma_s"]
            alpha_s, beta_s = self.transform.transform_params(mu_s, sigma_s)
        elif all([p is not None for p in p1]):
            # User provided `alpha` and `beta`
            pass
        elif all([p is not None for p in p2]):
            # User provided `mu` and `sigma`
            alpha_s, beta_s = self.transform.transform_params(mu_s, sigma_s)
        else:
            raise ValueError("invalid parameter combination")
        alpha_s = cast(alpha_s)
        beta_s = cast(beta_s)
        self.q, _, _, self.Q, _, _ = self._integral_op(alpha_s, beta_s)
        self.eigQ = eigen(self.Q)

    def _first_moment(self, e=None):
        return self.q

    def _second_moment(self, eigE=None):
        return self.eigQ
