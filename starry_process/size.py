from .integrals import MomentIntegral
from .ops import CheckBoundsOp
import theano.tensor as tt
import numpy as np
from scipy.special import legendre as P
from theano.ifelse import ifelse
from .math import matrix_sqrt


class Spot:
    def __init__(
        self,
        ydeg=15,
        spts=1000,
        eps4=1e-9,
        smoothing=0.075,
        sfac=300,
        cutoff=1.5,
        **kwargs
    ):
        """
        TODO: Expose some of these kwargs to the user?

        """
        theta = np.linspace(0, np.pi, spts)
        cost = np.cos(theta)
        B = np.hstack(
            [
                np.sqrt(2 * l + 1) * P(l)(cost).reshape(-1, 1)
                for l in range(ydeg + 1)
            ]
        )
        A = np.linalg.solve(B.T @ B + eps4 * np.eye(ydeg + 1), B.T)
        l = np.arange(ydeg + 1)
        i = l * (l + 1)
        S = np.exp(-0.5 * i * smoothing ** 2)
        Bp = S[:, None] * A
        self.i = i
        self.ij = np.ix_(i, i)
        self.N = (ydeg + 1) ** 2
        self.theta = tt.as_tensor_variable(theta)
        self.Bp = tt.as_tensor_variable(Bp)
        self.sfac = sfac
        self.cutoff = cutoff

    def b(self, r):
        z = self.sfac * (self.theta - r)
        return 1 / (1 + tt.exp(-z)) - 1

    def get_y(self, r):
        b = self.b(r)
        y = tt.zeros(self.N)
        y = tt.set_subtensor(y[self.i], tt.dot(self.Bp, b))
        return y

    def get_e(self, r, d):
        chim = tt.exp(self.sfac * (r - d - self.theta))
        chip = tt.exp(self.sfac * (r + d - self.theta))
        c = 1.0 / (2 * d * self.sfac) * tt.log((1 + chim) / (1 + chip))
        e = tt.zeros(self.N)
        e = tt.set_subtensor(e[self.i], tt.dot(self.Bp, c))
        return e

    def get_eigE(self, r, d):
        # NOTE: For theta > r + d, the `C` matrix drops
        # to zero VERY quickly. In practice we get better
        # numerical stability if we just set those elements
        # to zero without evaluating them, especially since
        # we can get NaNs from operations involving the extremely
        # large dynamic range between the `exp` and `ln` terms.
        # TODO: Find a more numerically stable way of evaluating this.
        kmax = tt.argmax(self.theta / (r + d) > self.cutoff)
        t = tt.reshape(self.theta[:kmax], (1, -1))
        chim = tt.exp(self.sfac * (r - d - t))
        chip = tt.exp(self.sfac * (r + d - t))
        exp = tt.exp(self.sfac * (t - tt.transpose(t)))
        term = tt.log(1 + chim) - tt.log(1 + chip)
        C0 = (exp * term - tt.transpose(term)) / (1 - exp + 1.0e-15)
        C0 = tt.set_subtensor(
            C0[tt.arange(kmax), tt.arange(kmax)],
            (
                tt.reshape(
                    1 / (1 + chip) + chim / (1 + chim) - term - 1, (-1,),
                )
            ),
        )
        C0 /= 2 * d * self.sfac
        C = tt.zeros((self.theta.shape[0], self.theta.shape[0]))
        C = tt.set_subtensor(C[:kmax, :kmax], C0)
        Etilde = tt.dot(tt.dot(self.Bp, C), tt.transpose(self.Bp))
        eigEtilde = matrix_sqrt(Etilde)
        eigE = tt.zeros((self.N, self.N))
        eigE = tt.set_subtensor(eigE[self.ij], eigEtilde)
        return eigE


class SizeIntegral(MomentIntegral):
    def _ingest(self, r, d, **kwargs):
        """
        Ingest the parameters of the distribution and 
        set up the transform and rotation operators.

        """
        # Ingest it
        self._r = CheckBoundsOp(name="r", lower=0, upper=0.5 * np.pi)(
            r * self._angle_fac
        )
        self._d = CheckBoundsOp(name="d", lower=0, upper=0.5 * np.pi)(
            d * self._angle_fac
        )
        self._params = [self._r, self._d]

        # Set up the spot operator
        self._spot = Spot(ydeg=self._ydeg, **kwargs)

        # Compute the moments
        self._q = ifelse(
            tt.eq(self._d, 0),
            self._spot.get_y(self._r),
            self._spot.get_e(self._r, self._d),
        )
        self._eigQ = ifelse(
            tt.eq(self._d, 0),
            tt.set_subtensor(tt.zeros((self._nylm, 2))[:, 0], self._q),
            self._spot.get_eigE(self._r, self._d),
        )

    def _compute(self):
        pass

    def _first_moment(self, e=None):
        return self._q

    def _second_moment(self, eigE=None):
        return self._eigQ
