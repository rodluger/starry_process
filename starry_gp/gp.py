from .size import SizeIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
from .transform import eigen
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class YlmGP(object):
    def __init__(self, ydeg, **kwargs):
        assert ydeg > 0, "Degree of map must be > 0."
        self.ydeg = ydeg
        self.S = SizeIntegral(ydeg)
        self.P = LatitudeIntegral(ydeg)
        self.L = LongitudeIntegral(ydeg)
        self.set_params(**kwargs)

    def set_params(
        self,
        mu_lat=0.5,
        nu_lat=0.01,
        mu_r=0.025,
        nu_r=0.005,
        mu_b=np.log(0.3),
        nu_b=1.0,
        **kwargs,
    ):
        # Set params for each integral
        self.S.set_params(mu_r=mu_r, nu_r=nu_r, mu_b=mu_b, nu_b=nu_b)
        self.P.set_params(mu_lat=mu_lat, nu_lat=nu_lat)

        # Compute the mean
        vector = self.S.first_moment()
        vector = self.P.first_moment(vector)
        self.mu = self.L.first_moment(vector)

        # Compute the covariance
        matrix = self.S.second_moment()
        matrix = self.P.second_moment(matrix)

        # Lower the dimension of the decomposition
        # (NOT an approximation!)
        matrix = eigen(matrix @ matrix.T)

        matrix = self.L.second_moment(matrix)
        EyyT = matrix @ matrix.T
        Ey = self.mu.reshape(-1, 1)
        self.cov = EyyT - Ey @ Ey.T

    def draw(self, ndraws=1):
        npts = self.cov.shape[0]
        L = np.tril(cho_factor(self.cov, lower=True)[0])
        u = np.random.randn(npts, ndraws)
        x = np.dot(L, u) + self.mu[:, None]
        return x.T
