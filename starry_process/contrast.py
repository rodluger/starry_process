from .ops import CheckBoundsOp
from .defaults import defaults
import theano.tensor as tt
import numpy as np


class ContrastIntegral:
    def __init__(self, c, n, child=None, ydeg=defaults["ydeg"], **kwargs):
        assert child is not None
        self._child = child
        self._ydeg = ydeg
        self._nylm = (self._ydeg + 1) ** 2
        self._c = CheckBoundsOp(name="c", lower=0, upper=1)(c)
        self._n = CheckBoundsOp(name="n", lower=1, upper=np.inf)(n)

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
        eps1 = kwargs.pop("eps1", defaults["eps1"])
        eps2 = kwargs.pop("eps2", defaults["eps2"])
        lam = np.ones(self._nylm) * eps1
        lam[self._ydeg ** 2 :] = eps2
        lam = tt.diag(lam)
        self._cov += lam

    def mean(self):
        return self._mean

    def cov(self):
        return self._cov
