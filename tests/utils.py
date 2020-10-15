import theano.tensor as tt
import numpy as np


class DummyChild:
    def __init__(self, ydeg):
        self._ydeg = ydeg
        self._nylm = (ydeg + 1) ** 2

    def mean(self):
        return tt.as_tensor_variable(np.zeros(self._nylm))

    def cov(self):
        return tt.as_tensor_variable(np.zeros((self._nylm, self._nylm)))
