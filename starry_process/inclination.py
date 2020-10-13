from .wigner import R
from .integrals import WignerIntegral
from .transforms import FixedTransform, InclinationTransform
from .ops import RxOp, CheckBoundsOp
import numpy as np
from scipy.special import gamma, hyp2f1


class InclinationIntegral(WignerIntegral):
    def _ingest(self, params, **kwargs):
        """
        Ingest the parameters of the distribution and 
        set up the transform and rotation operators.

        """
        if not hasattr(params, "__len__"):
            if params is None:
                params = []
            else:
                params = [params]

        if len(params) == 0:

            # The distribution is just the sine distribution over [0, pi/2]
            self._fixed = False
            self._params = []

            # Set up the transform
            self._transform = InclinationTransform()

            # Set up the rotation operator
            self._R = R(
                self._ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
            )

            """
            # Old parametrization: rotate from equatorial frame
            G = lambda i, j: (
                gamma(0.5 * (i + 1))
                * gamma(0.5 * (j + 1))
                / gamma(0.5 * (2 + i + j))
                - 2 ** (0.5 * (1 - i))
                / (i + 1)
                * hyp2f1(0.5 * (i + 1), 0.5 * (1 - j), 0.5 * (i + 3), 0.5)
            )
            term = np.zeros((4 * self._ydeg + 1, 4 * self._ydeg + 1))
            for i in range(4 * self._ydeg + 1):
                for j in range(0, 4 * self._ydeg + 1):
                    term[i, j] = G(i + 2, j) - G(i, j + 2)
            """

            # Integrate the basis terms in the *polar frame*
            # This is the integral of
            #
            #    cos(x / 2)^i sin(x / 2)^j sin(x)
            #
            # from 0 to pi/2.
            term = np.zeros((4 * self._ydeg + 1, 4 * self._ydeg + 1))
            for i in range(4 * self._ydeg + 1):
                for j in range(0, 4 * self._ydeg + 1):
                    term[i, j] = 2 * gamma(1 + 0.5 * i) * gamma(
                        1 + 0.5 * j
                    ) / gamma(0.5 * (4 + i + j)) - (
                        2 ** (1 - 0.5 * i) / (2 + i)
                    ) * hyp2f1(
                        1 + 0.5 * i, -0.5 * j, 2 + 0.5 * i, 0.5
                    )

            # Compute the moment integrals
            self._q = np.zeros(self._nylm)
            self._Q = np.zeros((self._nylm, self._nylm))
            n1 = 0
            for l1 in range(self._ydeg + 1):
                for m1 in range(-l1, l1 + 1):
                    j1 = m1 + l1
                    i1 = l1 - m1
                    self._q[n1] = term[j1, i1]
                    n2 = 0
                    for l2 in range(self._ydeg + 1):
                        for m2 in range(-l2, l2 + 1):
                            j2 = m2 + l2
                            i2 = l2 - m2
                            self._Q[n1, n2] = term[j1 + j2, i1 + i2]
                            n2 += 1
                    n1 += 1

        elif len(params) == 1:

            # The user passed in a *constant* inclination value
            self._fixed = True
            self._params = [
                CheckBoundsOp(name="value", lower=0, upper=0.5 * np.pi)(
                    params[0] * self._angle_fac
                )
            ]

            # Set up the transform
            self._transform = FixedTransform()

            # Set up the rotation operator
            self._Rx = RxOp(self._ydeg, **kwargs)

        else:

            raise ValueError("Invalid number of parameters.")

    def _dotRx(self, M, theta):
        """
        Dot a matrix `M` into the Wigner x-hat rotation matrix `Rx`.

        """
        f = tt.zeros_like(M)
        rx = self._Rx(theta)[0]
        nwig = lambda l: ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3
        for l in range(self.ydeg + 1):
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
        # TODO: Check the sign
        inc = self._params[0]
        if M.ndim == 1:
            return tt.reshape(self._dotRx(tt.reshape(M, (1, -1)), -inc))
        else:
            return tt.transpose(self._dotRx(tt.transpose(MT), -inc))
