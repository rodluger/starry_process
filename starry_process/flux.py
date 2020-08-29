# -*- coding: utf-8 -*-
from .ops import RxOp, tensordotRzOp, rTA1Op
import theano.tensor as tt
import numpy as np

__all__ = ["FluxDesignMatrix"]


class FluxDesignMatrix(object):
    def __init__(self, ydeg, **kwargs):
        self.ydeg = ydeg
        self.Rx = RxOp(ydeg, **kwargs)
        self.tensordotRz = tensordotRzOp(ydeg, **kwargs)
        self.rTA1 = rTA1Op(ydeg, **kwargs)()

    def _nwig(self, l):
        return ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3

    def dotRx(self, M, theta):
        f = tt.zeros_like(M)
        rx = self.Rx(theta)[0]
        for l in range(self.ydeg + 1):
            start = self._nwig(l - 1)
            stop = self._nwig(l)
            Rxl = tt.reshape(rx[start:stop], (2 * l + 1, 2 * l + 1))
            f = tt.set_subtensor(
                f[:, l ** 2 : (l + 1) ** 2],
                tt.dot(M[:, l ** 2 : (l + 1) ** 2], Rxl),
            )
        return f

    def right_project(self, M, theta, inc):
        """Apply the projection operator on the right.

        Specifically, this method returns the dot product `M . R`,
        where `M` is an input matrix and R` is the Wigner rotation matrix
        that transforms a spherical harmonic coefficient vector in the
        input frame to a vector in the observer's frame.
        """
        # Rotate to the sky frame
        M = self.dotRx(M, -inc)

        # Rotate to the correct phase
        M = self.tensordotRz(M, theta)

        # Rotate to the polar frame
        M = self.dotRx(M, 0.5 * np.pi)

        return M

    def __call__(self, theta, inc):
        rTA1 = tt.tile(self.rTA1, (theta.shape[0], 1))
        X = self.right_project(rTA1, theta, inc)
        return X
