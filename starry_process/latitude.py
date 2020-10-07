from .wigner import R
from .integrals import WignerIntegral
from .transforms import FixedTransform, LatitudeTransform
from .ops import RxOp, LatitudeIntegralOp, CheckBoundsOp
from .math import cast
from .defaults import defaults
import theano.tensor as tt
import numpy as np


class LatitudeIntegral(WignerIntegral):
    def _ingest(self, params, **kwargs):
        """
        Ingest the parameters of the distribution and 
        set up the transform and rotation operators.

        """
        if not hasattr(params, "__len__"):
            params = [params]

        if len(params) == 1:

            # User passed the *constant* latitude value
            self._fixed = True

            # Ingest it
            self._params = [
                CheckBoundsOp(name="value", lower=0, upper=0.5 * np.pi)(
                    params[0] * self._angle_fac
                )
            ]

            # Set up the transform
            self._transform = FixedTransform()

            # Set up the rotation operator
            self._Rx = RxOp(self._ydeg, **kwargs)

        elif len(params) == 2:

            # User passed `a`, `b` characterizing the latitude distribution
            self._fixed = False

            # Ingest them
            self._params = [
                CheckBoundsOp(name="a", lower=0, upper=1)(params[0]),
                CheckBoundsOp(name="b", lower=0, upper=1)(params[1]),
            ]

            # Set up the transform
            self._transform = LatitudeTransform(**kwargs)

            # Set up the rotation operator
            self._R = R(
                self._ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
            )

            # Compute the integrals
            self._integral_op = LatitudeIntegralOp(self._ydeg, **kwargs)
            alpha, beta = self._transform._ab_to_alphabeta(*self._params)
            self._q, _, _, self._Q, _, _ = self._integral_op(alpha, beta)

        else:

            raise ValueError("Invalid number of parameters.")

    def _dotRx(self, M, theta):
        """
        Dot a matrix `M` into the Wigner x-hat rotation matrix `Rx`.

        """
        f = tt.zeros_like(M)
        rx = self._Rx(theta)[0]
        nwig = lambda l: ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3
        for l in range(self._ydeg + 1):
            start = nwig(l - 1)
            stop = nwig(l)
            Rxl = tt.reshape(rx[start:stop], (2 * l + 1, 2 * l + 1))
            f = tt.set_subtensor(
                f[:, l ** 2 : (l + 1) ** 2],
                tt.dot(M[:, l ** 2 : (l + 1) ** 2], Rxl),
            )
        return f

    def _rotate(self, M):
        """
        Rotate a matrix `M` about the `x` axis.
        
        """
        phi = self._params[0]
        if M.ndim == 1:
            MT = 0.5 * tt.reshape(M, (1, -1))
            return tt.reshape(
                self._dotRx(MT, phi) + self._dotRx(MT, -phi), (-1,),
            )
        else:
            MT = 0.5 * tt.transpose(M)
            return tt.transpose(self._dotRx(MT, phi) + self._dotRx(MT, -phi))
