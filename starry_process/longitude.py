from .wigner import R
from .integrals import WignerIntegral
from .ops import RyOp, CheckBoundsOp
import numpy as np
from scipy.special import gamma


class LongitudeIntegral(WignerIntegral):
    def _ingest(self, **kwargs):
        """
        Set up the transform and rotation operators.

        """
        # Set up the transform
        self._params = []

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

    def _pdf(self, lam):
        """
        Return the probability density function evaluated at a 
        longitude `lam`.
        
        .. note:: 
        
            This function operates on and returns numeric values.
            It is used internally in the `perform` step of a `PDFOp`.

        """
        return np.ones_like(lam) / 2 * np.pi * self._angle_fac

    def _sample(self, nsamples=1):
        """
        Draw samples from the latitude distribution (in degrees).

        .. note:: 
        
            This function operates on and returns numeric values.
            It is used internally in the `perform` step of a `SampleOp`.
        
        """
        return (
            2
            * np.pi
            * (np.random.random(size=nsamples) - 0.5)
            / self._angle_fac
        )
