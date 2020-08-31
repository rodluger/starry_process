import numpy as np
from scipy.special import hyp2f1


class LatitudeIntegralOp(object):
    def __init__(self, ydeg, **kwargs):
        self.ydeg = ydeg
        self.N = (self.ydeg + 1) ** 2

    def __call__(self, alpha, beta):
        # Init
        n = 4 * self.ydeg + 1
        B = np.zeros(n)
        F = np.zeros(n)
        q = np.zeros(self.N)
        Q = np.zeros((self.N, self.N))

        # B functions
        B[0] = 1.0
        for k in range(1, n):
            B[k] = (alpha - 1.0 + k) / (alpha + beta - 1.0 + k) * B[k - 1]

        # F functions
        ab = alpha + beta
        F[0] = np.sqrt(2.0) * hyp2f1(-0.5, beta, ab, 0.5)
        F[1] = np.sqrt(2.0) * hyp2f1(-0.5, beta, ab + 1.0, 0.5)
        for k in range(2, n):
            F[k] = (
                (ab + k - 1.0)
                / ((alpha + k - 1.0) * (ab + k - 0.5))
                * ((ab + k - 2.0) * F[k - 2] + (1.5 - beta) * F[k - 1])
            )
        F *= B

        # Terms
        term = np.zeros((n, n))
        for i in range(n):
            if i % 2 == 0:
                func = B
                i2 = i // 2
            else:
                func = F
                i2 = (i - 1) // 2

            for j in range(0, n, 2):
                j2 = j // 2
                fac1 = 1.0
                for k1 in range(i2 + 1):
                    fac2 = fac1
                    for k2 in range(j2 + 1):
                        term[i, j] += fac2 * func[k1 + k2]
                        fac2 *= (k2 - j2) / (k2 + 1.0)
                    fac1 *= (i2 - k1) / (k1 + 1.0)

        # Beta normalization
        term /= B[0]

        # Moment integrals
        n1 = 0
        inv_two_l1 = 1.0
        for l1 in range(self.ydeg + 1):
            for m1 in range(-l1, l1 + 1):
                j1 = m1 + l1
                i1 = l1 - m1
                q[n1] = term[j1, i1] * inv_two_l1
                n2 = 0
                inv_two_l1l2 = inv_two_l1
                for l2 in range(self.ydeg + 1):
                    for m2 in range(-l2, l2 + 1):
                        j2 = m2 + l2
                        i2 = l2 - m2
                        Q[n1, n2] = term[j1 + j2, i1 + i2] * inv_two_l1l2
                        n2 += 1
                    inv_two_l1l2 *= 0.5
                n1 += 1
            inv_two_l1 *= 0.5

        return q, None, None, Q, None, None

