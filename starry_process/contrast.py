from .ops import CheckBoundsOp
from .defaults import defaults
from .math import cast
from .compat import tt
import numpy as np


class ContrastIntegral:
    def __init__(self, c, n, child=None, ydeg=defaults["ydeg"], **kwargs):
        assert child is not None
        self._child = child
        self._ydeg = ydeg
        self._nylm = (self._ydeg + 1) ** 2
        self._c = cast(c)
        self._n = CheckBoundsOp(name="n", lower=0, upper=np.inf)(n)

        # Compute the full Ylm mean and covariance weighted by
        # the spot contrast & number of spots
        mom1 = self._child.first_moment()
        eig_mom2 = self._child.second_moment()
        mom2 = tt.dot(eig_mom2, tt.transpose(eig_mom2))
        self._mean = np.pi * self._c * self._n * mom1
        self._cov = (
            (np.pi * self._c) ** 2 * self._n * (mom2 - tt.outer(mom1, mom1))
        )

        # Stability hacks
        epsy = kwargs.pop("epsy", defaults["epsy"])
        epsy15 = kwargs.pop("epsy15", defaults["epsy15"])
        lam = np.ones(self._nylm) * epsy
        lam[15 ** 2 :] = epsy15
        lam = tt.diag(lam)
        self._cov += lam

    def mean(self):
        return self._mean

    def cov(self):
        return self._cov
