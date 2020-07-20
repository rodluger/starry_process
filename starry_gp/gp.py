from .spot import SpotIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
from .transform import eigen
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class YlmGP(object):
    def __init__(self, ydeg, **kwargs):
        assert ydeg > 0, "Degree of map must be > 0."
        self.ydeg = ydeg
        self.S = SpotIntegral(ydeg)
        self.P = LatitudeIntegral(ydeg)
        self.L = LongitudeIntegral(ydeg)
        self.set_params(**kwargs)

    def set_params(
        self,
        mu_beta=0.5,
        nu_beta=0.01,
        mu_lns=np.log(0.05),
        sig_lns=0.1,
        mu_lna=np.log(0.01),
        sig_lna=0.1,
        sign=-1,
        **kwargs
    ):
        # Set params for each integral
        self.S.set_params(
            mu_lns=mu_lns, sig_lns=sig_lns, mu_lna=mu_lna, sig_lna=sig_lna, sign=sign
        )
        self.P.set_params(mu_beta=mu_beta, nu_beta=nu_beta)

        # Compute the mean
        vector = self.S.first_moment()
        vector = self.P.first_moment(vector)
        self.mu = self.L.first_moment(vector)

        # Compute the covariance
        matrix = self.S.second_moment()
        matrix = self.P.second_moment(matrix)

        # Trick to lower the size of the matrix
        # (NOT an approximation!)
        matrix = eigen(matrix @ matrix.T)

        matrix = self.L.second_moment(matrix)
        EyyT = matrix @ matrix.T
        Ey = self.mu.reshape(-1, 1)
        self.cov = EyyT - Ey @ Ey.T

        # Discard the constant term (=0)
        self.mu = self.mu[1:]
        self.cov = self.cov[1:, 1:]

    def draw(self, ndraws=1):
        npts = self.cov.shape[0]
        L = np.tril(cho_factor(self.cov, lower=True)[0])
        u = np.random.randn(npts, ndraws)
        x = np.dot(L, u) + self.mu[:, None]
        return x.T
