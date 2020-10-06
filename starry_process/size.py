from .integrals import MomentIntegral
from .transforms import SizeTransform
from .math import cast, matrix_sqrt
from .ops import SizeIntegralOp, CheckBoundsOp
from .defaults import defaults
import theano.tensor as tt
import numpy as np
from scipy.special import legendre as P


class DiscreteSpot(object):
    def __init__(
        self, ydeg=15, npts=1000, eps=1e-9, smoothing=0.075, sfac=300, **kwargs
    ):
        theta = np.linspace(0, np.pi, npts)
        cost = np.cos(theta)
        B = np.hstack(
            [
                np.sqrt(2 * l + 1) * P(l)(cost).reshape(-1, 1)
                for l in range(ydeg + 1)
            ]
        )
        A = np.linalg.solve(B.T @ B + eps * np.eye(ydeg + 1), B.T)
        l = np.arange(ydeg + 1)
        i = l * (l + 1)
        S = np.exp(-0.5 * i * smoothing ** 2)
        A = S[:, None] * A
        self.i = i
        self.N = (ydeg + 1) ** 2
        self.theta = tt.as_tensor_variable(theta)
        self.A = tt.as_tensor_variable(A)
        self.sfac = sfac

    def S(self, s):
        z = self.sfac * (self.theta - s)
        return 1 / (1 + tt.exp(-z)) - 1

    def get_y(self, s):
        I = self.S(s)
        y = tt.zeros(self.N)
        y = tt.set_subtensor(y[self.i], tt.dot(self.A, I))
        return y


class SizeIntegral(MomentIntegral):
    def _precompute(
        self,
        fix_size=defaults["fix_size"],
        angle_unit=defaults["angle_unit"],
        **kwargs
    ):
        self.fixed = fix_size
        if angle_unit.startswith("deg"):
            self._angle_fac = np.pi / 180
        elif angle_unit.startswith("rad"):
            self._angle_fac = 1.0
        else:
            raise ValueError("Invalid `angle_unit`.")
        if self.fixed:
            self.transform = None
            self.check_bounds_s = CheckBoundsOp(
                name="sa", lower=0, upper=0.5 * np.pi
            )
            self.spot = DiscreteSpot(ydeg=self.ydeg, **kwargs)
        else:
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

    def _set_params(
        self, s=defaults["s"], sa=defaults["sa"], sb=defaults["sb"], **kwargs
    ):
        if self.fixed:
            self.s = self.check_bounds_s(s * self._angle_fac)
            self.q = self.spot.get_y(self.s)
            self.eigQ = tt.reshape(self.q, (-1, 1))
        else:
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
        if self.fixed:
            return 0.0
        else:
            return self.transform.log_jac(self.sa, self.sb)
