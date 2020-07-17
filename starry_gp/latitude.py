from .transform import TransformIntegral
import numpy as np
from scipy.special import gamma, hyp2f1


class LatitudeIntegral(TransformIntegral):
    def __init__(self, ydeg):
        super().__init__(ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1)
        self.set_params()

    def _compute_basis_integrals(self, alpha=2.0, beta=2.0):

        # Integrate the basis terms

        # B functions
        B = np.zeros(4 * self.ydeg + 1)
        B[0] = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
        for n in range(1, 4 * self.ydeg + 1):
            B[n] = (alpha - 1 + n) / (alpha + beta - 1 + n) * B[n - 1]

        # F functions
        ab = alpha + beta
        F = np.zeros(4 * self.ydeg + 1)
        F[0] = hyp2f1(-0.5, alpha, ab, -1)
        F[1] = hyp2f1(-0.5, alpha + 1, ab + 1, -1)
        for n in range(2, 4 * self.ydeg + 1):
            F[n] = (
                (ab + n - 1)
                / ((alpha + n - 1) * (ab + n - 0.5))
                * ((ab + n - 2) * F[n - 2] + (1.5 - beta) * F[n - 1])
            )
        F *= B

        # Terms
        term = np.zeros((4 * self.ydeg + 1, 4 * self.ydeg + 1))
        for i in range(4 * self.ydeg + 1):
            if i % 2 == 0:
                func = B
            else:
                func = F
            i2 = i // 2
            for j in range(0, 4 * self.ydeg + 1, 2):
                j2 = j // 2
                fac1 = 1.0
                for k1 in range(i2 + 1):
                    fac2 = fac1
                    for k2 in range(j2 + 1):
                        term[i, j] += fac2 * func[k1 + k2]
                        fac2 *= (k2 - j2) / (k2 + 1.0)
                    fac1 *= (i2 - k1) / (k1 + 1)

        # Beta normalization
        term /= B[0]

        # Moment integrals
        self.q = np.zeros(self.N)
        self.Q = np.zeros((self.N, self.N))
        n1 = 0
        for l1 in range(self.ydeg + 1):
            for m1 in range(-l1, l1 + 1):
                j1 = m1 + l1
                i1 = l1 - m1
                self.q[n1] = term[j1, i1] / (2 ** l1)
                n2 = 0
                for l2 in range(self.ydeg + 1):
                    for m2 in range(-l2, l2 + 1):
                        j2 = m2 + l2
                        i2 = l2 - m2
                        self.Q[n1, n2] = term[j1 + j2, i1 + i2] / (2 ** (l1 + l2))
                        n2 += 1
                n1 += 1
