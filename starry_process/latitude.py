from .wigner import R
from .integrals import WignerIntegral
from .transforms import LatitudeTransform
from .ops import RxOp, LatitudeIntegralOp, CheckBoundsOp
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
            self._Rx = RxOp(self.ydeg, **kwargs)
            self.check_bounds_l = CheckBoundsOp(
                name="l", lower=0, upper=0.5 * np.pi
            )
        else:
            self.transform = LatitudeTransform(**kwargs)
            self.R = R(
                self.ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
            )
            self._integral_op = LatitudeIntegralOp(self.ydeg, **kwargs)
            self.check_bounds_la = CheckBoundsOp(name="la", lower=0, upper=1)
            self.check_bounds_lb = CheckBoundsOp(name="lb", lower=0, upper=1)

    def _compute_basis_integrals(
        self, la=defaults["la"], lb=defaults["lb"], **kwargs
    ):
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

    def _set_params(self, l=defaults["l"], **kwargs):
        if self.fixed:
            self.l = self.check_bounds_l(l * self._angle_fac)
        else:
            super()._set_params(**kwargs)

    def _nwig(self, l):
        return ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3

    def _dotRx(self, M, theta):
        f = tt.zeros_like(M)
        rx = self._Rx(theta)[0]
        for l in range(self.ydeg + 1):
            start = self._nwig(l - 1)
            stop = self._nwig(l)
            Rxl = tt.reshape(rx[start:stop], (2 * l + 1, 2 * l + 1))
            f = tt.set_subtensor(
                f[:, l ** 2 : (l + 1) ** 2],
                tt.dot(M[:, l ** 2 : (l + 1) ** 2], Rxl),
            )
        return f

    def _first_moment(self, e):
        if self.fixed:
            eT = 0.5 * tt.reshape(e, (1, -1))
            return tt.reshape(
                self._dotRx(eT, self.l) + self._dotRx(eT, -self.l), (-1,),
            )
        else:
            return super()._first_moment(e)

    def _second_moment(self, eigE):
        if self.fixed:
            eigET = 0.5 * tt.transpose(eigE)
            return tt.transpose(
                self._dotRx(eigET, self.l) + self._dotRx(eigET, -self.l)
            )
        else:
            return super()._second_moment(eigE)
