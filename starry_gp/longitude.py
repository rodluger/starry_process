from .wigner import R
from .integrals import WignerIntegral
import numpy as np
from scipy.special import gamma


class LongitudeIntegral(WignerIntegral):
    def _precompute(self, **kwargs):
        self.R = R(
            self.ydeg, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0
        )
        self.set_params()

    def _compute_basis_integrals(self):

        # Integrate the basis terms
        term = np.zeros((4 * self.ydeg + 1, 4 * self.ydeg + 1))
        for i in range(4 * self.ydeg + 1):
            for j in range(0, 4 * self.ydeg + 1, 2):
                term[i, j] = (
                    gamma(0.5 * (i + 1))
                    * gamma(0.5 * (j + 1))
                    / gamma(0.5 * (2 + i + j))
                )
        term /= np.pi

        # Moment integrals
        self.q = np.zeros(self.N)
        self.Q = np.zeros((self.N, self.N))
        n1 = 0
        for l1 in range(self.ydeg + 1):
            for m1 in range(-l1, l1 + 1):
                j1 = m1 + l1
                i1 = l1 - m1
                self.q[n1] = term[j1, i1]
                n2 = 0
                for l2 in range(self.ydeg + 1):
                    for m2 in range(-l2, l2 + 1):
                        j2 = m2 + l2
                        i2 = l2 - m2
                        self.Q[n1, n2] = term[j1 + j2, i1 + i2]
                        n2 += 1
                n1 += 1
