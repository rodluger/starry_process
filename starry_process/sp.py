from .size import SizeIntegral
from .contrast import ContrastIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
from .flux import FluxDesignMatrix
from .math import cho_factor
import theano.tensor as tt
from theano.tensor.shared_randomstreams import RandomStreams


class StarryProcess(object):
    def __init__(self, ydeg, **kwargs):
        assert ydeg > 0, "Degree of map must be > 0."
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2

        self.contrast = ContrastIntegral(self, **kwargs)
        self.longitude = LongitudeIntegral(self.contrast, **kwargs)
        self.latitude = LatitudeIntegral(self.longitude, **kwargs)
        self.size = SizeIntegral(self.latitude, **kwargs)
        self.design = FluxDesignMatrix(self.ydeg)

        self._mean_ylm = None
        self._cov_ylm = None
        self._cho_cov_ylm = None

        self._srng = RandomStreams(seed=0)

    @property
    def mean_ylm(self):
        if (self._mean_ylm is None) or (self.contrast.e is None):
            self._mean_ylm = self.contrast.first_moment()
        return self._mean_ylm

    def mean(self, t):
        return tt.dot(self.design(t), self.mean_ylm)

    @property
    def cov_ylm(self):
        if (self._cov_ylm is None) or (self.contrast.eigE is None):
            e4 = self.contrast.first_moment()
            eigE4 = self.contrast.second_moment()
            self._cov_ylm = tt.dot(eigE4, eigE4.T) - tt.outer(e4, e4)
            self._cho_cov_ylm = None
        return self._cov_ylm

    def cov(self, t):
        A = self.design(t)
        return tt.dot(tt.dot(A, self.cov_ylm), tt.transpose(A))

    def draw_ylm(self, ndraws=1, seed=None, eps=1e-12):
        if self._cho_cov_ylm is None:
            self._cho_cov_ylm = cho_factor(self.cov_ylm + tt.eye(self.N) * eps)
        if seed is not None:
            self._srng.seed(seed)
        u = self._srng.normal((self.N, ndraws))
        return tt.transpose(
            self.mean_ylm[:, None] + tt.dot(self._cho_cov_ylm, u)
        )

    def draw(self, t, ndraws=1, seed=None, eps=1e-12):
        ylm = self.draw_ylm(ndraws=ndraws, seed=seed, eps=eps)
        return tt.dot(ylm, tt.transpose(self.design(t)))
