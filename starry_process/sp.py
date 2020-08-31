from .size import SizeIntegral
from .contrast import ContrastIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
from .flux import FluxDesignMatrix
from .math import theano_math, numpy_math


class StarryProcess(object):
    def __init__(self, ydeg, **kwargs):
        assert ydeg > 0, "Degree of map must be > 0."
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2
        self._math = (
            theano_math if kwargs.get("use_theano", True) else numpy_math
        )

        self.contrast = ContrastIntegral(self, **kwargs)
        self.longitude = LongitudeIntegral(self.contrast, **kwargs)
        self.latitude = LatitudeIntegral(self.longitude, **kwargs)
        self.size = SizeIntegral(self.latitude, **kwargs)
        self.design = FluxDesignMatrix(self.ydeg)

        self._mean_ylm = None
        self._cov_ylm = None
        self._cho_cov_ylm = None

        # NB: Change this by setting `self.random.seed(XXX)`
        self.random = self._math.RandomStreams(seed=0)

    @property
    def mean_ylm(self):
        if (self._mean_ylm is None) or (self.contrast.e is None):
            self._mean_ylm = self.contrast.first_moment()
        return self._mean_ylm

    def mean(self, t):
        return self._math.dot(self.design(t), self.mean_ylm)

    @property
    def cov_ylm(self):
        if (self._cov_ylm is None) or (self.contrast.eigE is None):
            e4 = self.contrast.first_moment()
            eigE4 = self.contrast.second_moment()
            self._cov_ylm = self._math.dot(
                eigE4, self._math.transpose(eigE4)
            ) - self._math.outer(e4, e4)
            self._cho_cov_ylm = None
        return self._cov_ylm

    def cov(self, t):
        A = self.design(t)
        return self._math.dot(
            self._math.dot(A, self.cov_ylm), self._math.transpose(A)
        )

    def draw_ylm(self, ndraws=1, eps=1e-12):
        if (
            (self.contrast.eigE is None)
            or (self._cov_ylm is None)
            or (self._cho_cov_ylm is None)
        ):
            self._cho_cov_ylm = self._math.cho_factor(
                self.cov_ylm + self._math.eye(self.N) * eps
            )
        u = self.random.normal((self.N, ndraws))
        return self._math.transpose(
            self.mean_ylm[:, None] + self._math.dot(self._cho_cov_ylm, u)
        )

    def draw(self, t, ndraws=1, eps=1e-12):
        ylm = self.draw_ylm(ndraws=ndraws, eps=eps)
        return self._math.dot(ylm, self._math.transpose(self.design(t)))
