from .transform import eigen
import numpy as np


class SpotIntegral(object):
    """Marginalizes over the spot size and amplitude distribution.

    Computes the first two moments of the distribution over spherical
    harmonic coefficients given a distribution of spot sizes and amplitudes.

    The spot size `sigma` is distributed according to

        ln_s ~ N(mu_lns, sig_lns)
    
    with

        sigma = exp(ln_s)

    The parameter `sigma` is the standard deviation of the two-dimensional
    Gaussian in the quantity `cos(theta)`, where `theta` is the angle between
    a point on the spot and the center of the spot, measured along the
    surface of the sphere.

    The spot amplitude is distributed according to

        ln_a ~ N(mu_lna, sig_lna)

    with

        amp = 1 / (1 + exp(-ln_a))

    The parameter `amp` is the integral over the intensity deficit (or excess)
    over the spot.

    Setting `sign = -1` (the default) results in dark spots; setting
    `sign = 1` results in bright spots.

    """

    def __init__(self, ydeg):
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2
        l = np.arange(ydeg + 1)
        self.i = l * (l + 1)
        self.ij = np.ix_(self.i, self.i)
        self.set_params()

        # Coefficients of the polynomial expansion of sigma for each Ylm
        self.coeffs = np.zeros((self.ydeg + 1, self.ydeg + 1))

        # Compute the integrals recursively
        IP = np.zeros((self.ydeg + 1, self.ydeg + 1))
        ID = np.zeros((self.ydeg + 1, self.ydeg + 1))

        # Seeding values
        IP[0, 0] = 1.0
        IP[1, 0] = 1.0
        IP[1, 1] = -np.sqrt(2 / np.pi)
        ID[1, 0] = 1.0
        self.coeffs[0] = 0  # IP[0]
        self.coeffs[1] = np.sqrt(3) * IP[1]

        # Recurse
        for n in range(2, self.ydeg + 1):
            C = 2.0 * n - 1.0
            A = C / n
            B = A - 1
            IP[n] = A * np.roll(ID[n - 1], 2) + A * IP[n - 1] - B * IP[n - 2]
            IP[n, 1] -= A * np.sqrt(2 / np.pi)
            ID[n] = C * IP[n - 1] + ID[n - 2]
            self.coeffs[n] = np.sqrt(2 * n + 1) * IP[n]

    def set_params(self, mu_lns=-3.0, sig_lns=0.1, mu_lna=-2.3, sig_lna=0.1, sign=-1):

        # Sigma Taylor basis
        n = np.arange(2 * self.ydeg + 1)
        tmp = np.exp(n * mu_lns + 0.5 * (n * sig_lns) ** 2)
        self.q = tmp[: self.ydeg + 1]
        self.Q = np.zeros((self.ydeg + 1, self.ydeg + 1))
        for n in range(self.ydeg + 1):
            for m in range(self.ydeg + 1):
                self.Q[n, m] = tmp[n + m]

        # Amplitude integrals
        self.q *= sign * np.exp(mu_lna + 0.5 * sig_lna ** 2)
        self.Q *= np.exp(2 * mu_lna + 2 * sig_lna ** 2)

        # Eigendecomposition of Q
        self.U = eigen(self.Q)

    def first_moment(self):
        """
        Returns the first moment `E[x]` of the spot size 
        and amplitude distribution.

        """
        S = np.zeros(self.N)
        S[self.i] = self.coeffs @ self.q
        return S

    def second_moment(self):
        """
        Returns the eigendecomposition `C` of the second moment `E[x^2]` 
        of the spot size and amplitude distribution, such that

            C @ C.T = E[x^2]

        """
        C = np.zeros((self.N, self.N))
        C[self.ij] = self.coeffs @ self.U
        return C
