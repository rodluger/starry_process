from .wigner import R
from .integrals import WignerIntegral
from .transforms import LatitudeTransform
from .ops import LatitudeIntegralOp, CheckBoundsOp
from .math import cast
from .defaults import defaults
import theano.tensor as tt
import numpy as np


class LatitudeIntegral(WignerIntegral):
    def _precompute(
        self,
        fix_latitude=defaults["fix_latitude"],
        angle_unit=defaults["angle_unit"],
        **kwargs
    ):
        self.fixed = fix_latitude
        if angle_unit.startswith("deg"):
            self._angle_fac = np.pi / 180
        elif angle_unit.startswith("rad"):
            self._angle_fac = 1.0
        else:
            raise ValueError("Invalid `angle_unit`.")
        if self.fixed:
            # TODO!
            raise NotImplementedError("Fixed latitude not yet implemented.")
        else:
            self.transform = LatitudeTransform(**kwargs)
            self.R = R(
                self.ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
            )
            self._integral_op = LatitudeIntegralOp(self.ydeg, **kwargs)
            self.check_bounds_la = CheckBoundsOp(name="la", lower=0, upper=1)
            self.check_bounds_lb = CheckBoundsOp(name="lb", lower=0, upper=1)

    def _compute_basis_integrals(
        self, l=defaults["l"], la=defaults["la"], lb=defaults["lb"], **kwargs
    ):
        if self.fixed:
            # TODO
            pass
        else:
            self.la = self.check_bounds_la(la)
            la1 = self.transform._ln_alpha_min
            la2 = self.transform._ln_alpha_max
            alpha_l = tt.exp(cast(la1 + self.la * (la2 - la1)))
            self.lb = self.check_bounds_lb(lb)
            lb1 = self.transform._ln_beta_min
            lb2 = self.transform._ln_beta_max
            beta_l = tt.exp(cast(lb1 + self.lb * (lb2 - lb1)))
            self.q, _, _, self.Q, _, _ = self._integral_op(alpha_l, beta_l)

    def _log_jac(self):
        if self.fixed:
            return 0.0
        else:
            return self.transform.log_jac(self.la, self.lb)
