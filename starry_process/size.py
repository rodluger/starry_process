from .integrals import MomentIntegral
from .transforms import FixedTransform, SizeTransform
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
    def _ingest(self, params, **kwargs):
        """
        Ingest the parameters of the distribution and 
        set up the transform and rotation operators.

        """
        if not hasattr(params, "__len__"):
            params = [params]

        if len(params) == 1:

            # User passed the *constant* size value
            self._fixed = True

            # Ingest it
            self._params = [
                CheckBoundsOp(name="value", lower=0, upper=0.5 * np.pi)(
                    params[0] * self._angle_fac
                )
            ]

            # Set up the transform
            self._transform = FixedTransform()

            # Set up the spot operator
            self._spot = DiscreteSpot(ydeg=self._ydeg, **kwargs)
            self._q = self._spot.get_y(self._params[0])
            self._eigQ = tt.reshape(self._q, (-1, 1))

        elif len(params) == 2:

            # User passed `a`, `b` characterizing the size distribution
            self._fixed = False

            # Ingest them
            self._params = [
                CheckBoundsOp(name="a", lower=0, upper=1)(params[0]),
                CheckBoundsOp(name="b", lower=0, upper=1)(params[1]),
            ]

            # Set up the transform
            self._transform = SizeTransform(ydeg=self._ydeg, **kwargs)

            # Compute the integrals
            kwargs.update(
                {
                    "compile_args": kwargs.get("compile_args", [])
                    + [
                        ("SP__C0", "{:.15f}".format(self._transform._c[0])),
                        ("SP__C1", "{:.15f}".format(self._transform._c[1])),
                        ("SP__C2", "{:.15f}".format(self._transform._c[2])),
                        ("SP__C3", "{:.15f}".format(self._transform._c[3])),
                    ]
                }
            )
            self._integral_op = SizeIntegralOp(self._ydeg, **kwargs)
            alpha, beta = self._transform._ab_to_alphabeta(*self._params)
            self._q, _, _, self._Q, _, _ = self._integral_op(alpha, beta)
            self._eigQ = matrix_sqrt(self._Q, driver=self._driver)

    def _compute(self):
        pass

    def _first_moment(self, e=None):
        return self._q

    def _second_moment(self, eigE=None):
        return self._eigQ
