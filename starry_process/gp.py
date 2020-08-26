from .size import SizeIntegral
from .contrast import ContrastIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
import theano.tensor as tt


class YlmGP(object):
    def __init__(self, ydeg, **kwargs):
        assert ydeg > 0, "Degree of map must be > 0."
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2

        self.contrast = ContrastIntegral(self, **kwargs)
        self.longitude = LongitudeIntegral(self.contrast, **kwargs)
        self.latitude = LatitudeIntegral(self.longitude, **kwargs)
        self.size = SizeIntegral(self.latitude, **kwargs)

        self._mean = None
        self._cov = None

    @property
    def mean(self):
        if (self._mean is None) or (self.contrast.e is None):
            self._mean = self.contrast.first_moment()
        return self._mean

    @property
    def cov(self):
        if (self._cov is None) or (self.contrast.eigE is None):
            e4 = self.contrast.first_moment()
            eigE4 = self.contrast.second_moment()
            self._cov = tt.dot(eigE4, eigE4.T) - tt.outer(e4, e4)
        return self._cov

    def draw(self, ndraws=1):
        # TODO
        raise NotImplementedError("TODO!")
        assert (
            self.mean is not None and self.cov is not None
        ), "Please call `compute` first."
        L = np.tril(cho_factor(self.cov, lower=True)[0])
        u = np.random.randn(self.N, ndraws)
        x = np.dot(L, u) + self.mean[:, None]
        return x.T
