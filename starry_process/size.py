from .integrals import MomentIntegral
from .ops import CheckBoundsOp
import theano.tensor as tt
import numpy as np
from scipy.special import legendre as P


class Spot:
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

    def S(self, r):
        z = self.sfac * (self.theta - r)
        return 1 / (1 + tt.exp(-z)) - 1

    def get_y(self, r):
        I = self.S(r)
        y = tt.zeros(self.N)
        y = tt.set_subtensor(y[self.i], tt.dot(self.A, I))
        return y


class SizeIntegral(MomentIntegral):
    def _ingest(self, r, **kwargs):
        """
        Ingest the parameters of the distribution and 
        set up the transform and rotation operators.

        """
        # Ingest it
        self._r = CheckBoundsOp(name="r", lower=0, upper=0.5 * np.pi)(
            r * self._angle_fac
        )
        self._params = [self._r]

        # Set up the spot operator
        self._spot = Spot(ydeg=self._ydeg, **kwargs)
        self._q = self._spot.get_y(self._r)
        self._eigQ = tt.reshape(self._q, (-1, 1))

    def _compute(self):
        pass

    def _first_moment(self, e=None):
        return self._q

    def _second_moment(self, eigE=None):
        return self._eigQ
