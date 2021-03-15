from .compat import tt
import numpy as np


__all__ = ["ExpSquaredKernel", "Matern32Kernel"]


def ExpSquaredKernel(t1, t2, tau):
    dt = tt.abs_(tt.reshape(t1, (-1, 1)) - tt.reshape(t2, (1, -1)))
    return tt.exp(-(dt ** 2) / (2 * tau))


def Matern32Kernel(t1, t2, tau):
    dt = tt.abs_(tt.reshape(t1, (-1, 1)) - tt.reshape(t2, (1, -1)))
    x = np.sqrt(3) * dt / tau
    return (1 + x) * tt.exp(-x)
