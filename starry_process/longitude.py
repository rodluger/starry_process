from .wigner import R
from .integrals import WignerIntegral
from .transforms import FixedTransform, LongitudeTransform
from .ops import RyOp, CheckBoundsOp
import numpy as np
from scipy.special import gamma


class LongitudeIntegral(WignerIntegral):
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

            # The distribution is just the uniform distribution over [0, 2pi]
            self._fixed = False
            self._params = []

            # Set up the transform
            self._transform = LongitudeTransform()

            # Set up the rotation operator
            self._R = R(
                self._ydeg, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0
            )

            # Integrate the basis terms
            term = np.zeros((4 * self._ydeg + 1, 4 * self._ydeg + 1))
            for i in range(4 * self._ydeg + 1):
                for j in range(0, 4 * self._ydeg + 1, 2):
                    term[i, j] = (
                        gamma(0.5 * (i + 1))
                        * gamma(0.5 * (j + 1))
                        / gamma(0.5 * (2 + i + j))
                    )
            term /= np.pi

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

            # The user passed in a *constant* longitude value
            self._fixed = True
            self._params = [
                CheckBoundsOp(name="value", lower=-np.pi, upper=np.pi)(
                    params[0] * self._angle_fac
                )
            ]

            # Set up the transform
            self._transform = FixedTransform()

            # Set up the rotation operator
            self._Ry = RyOp(self._ydeg, **kwargs)

            # TODO: The c code for the op above has not yet been implemented
            raise NotImplementedError(
                "Constant longitude not yet implemented!"
            )

        else:

            raise ValueError("Invalid number of parameters.")

    def _dotRy(self, M, theta):
        """
        Dot a matrix `M` into the Wigner y-hat rotation matrix `Ry`.

        """
        f = tt.zeros_like(M)
        rx = self._Ry(theta)[0]
        nwig = lambda l: ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3
        for l in range(self.ydeg + 1):
            start = nwig(l - 1)
            stop = nwig(l)
            Ryl = tt.reshape(rx[start:stop], (2 * l + 1, 2 * l + 1))
            f = tt.set_subtensor(
                f[:, l ** 2 : (l + 1) ** 2],
                tt.dot(M[:, l ** 2 : (l + 1) ** 2], Ryl),
            )
        return f

    def _rotate(self, M):
        """
        Rotate a matrix `M` about the `y` axis.
        
        """
        # TODO: check the sign
        lam = self._params[0]
        if M.ndim == 1:
            return tt.reshape(self._dotRy(tt.reshape(M, (1, -1)), -lam))
        else:
            return tt.transpose(self._dotRy(tt.transpose(MT), -lam))
