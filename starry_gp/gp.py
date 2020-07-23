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
        mu_lat=0.5,
        nu_lat=0.01,
        mu_s=0.1,
        nu_s=0.01,
        mu_a=-3.0,
        nu_a=1.0,
        **kwargs,
    ):
        # Set params for each integral
        self.S.set_params(mu_s=mu_s, nu_s=nu_s, mu_a=mu_a, nu_a=nu_a)
        self.P.set_params(mu_lat=mu_lat, nu_lat=nu_lat)

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
