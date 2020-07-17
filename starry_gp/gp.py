from .spot import SpotIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class YlmGP(object):
    def __init__(self, ydeg, **kwargs):
        self.ydeg = ydeg
        self.S = SpotIntegral(ydeg)
        self.P = LatitudeIntegral(ydeg)
        self.L = LongitudeIntegral(ydeg)
        self.set_params(**kwargs)

    def set_params(
        self,
        alpha=40.0,
        beta=20.0,
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
        self.P.set_params(alpha=alpha, beta=beta)

        # Compute the mean
        vector = self.S.first_moment()
        vector = self.P.first_moment(vector)
        self.mu = self.L.first_moment(vector)

        # Compute the covariance
        matrix = self.S.second_moment()
        matrix = self.P.second_moment(matrix)
        matrix = self.L.second_moment(matrix)
        EyyT = matrix @ matrix.T
        Ey = self.mu.reshape(-1, 1)
        self.cov = EyyT - Ey @ Ey.T

        if False:
            self._cho_cov = cho_factor(self.cov, lower=True)
            self._norm = -np.sum(np.log(np.diag(self._cho_cov[0]))) - 0.5 * len(
                self.mu
            ) * np.log(2 * np.pi)

    def log_likelihood(self, y):
        r = y - self.mu
        CInvy = cho_solve(self._cho_cov, y)
        return r.T.dot(CInvy) + self._norm

    def draw(self, ndraws=1):
        npts = self.cov.shape[0]
        L = np.tril(self._cho_cov[0])
        u = np.random.randn(npts, ndraws)
        x = np.dot(L, u) + self.mu[:, None]
        return x.T
