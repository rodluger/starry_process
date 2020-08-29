# -*- coding: utf-8 -*-
from .ops import RxOp, tensordotRzOp, rTA1Op
import theano.tensor as tt
import numpy as np

__all__ = ["FluxDesignMatrix"]


class FluxDesignMatrix(object):
    def __init__(self, ydeg, **kwargs):
        self.ydeg = ydeg
        self._Rx = RxOp(ydeg, **kwargs)
        self._tensordotRz = tensordotRzOp(ydeg, **kwargs)
        self._rTA1 = rTA1Op(ydeg, **kwargs)()
        self._set = False

    def set_params(self, t, period, inc):
        theta = 2 * np.pi / period * t
        inc *= np.pi / 180
        self._compute(theta, inc)
        self._set = True

    @property
    def A(self):
        assert self._set, "must call `set_params` first."
        return self._A

    @property
    def AT(self):
        assert self._set, "must call `set_params` first."
        return self._AT

    def _nwig(self, l):
        return ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3

    def _dotRx(self, M, theta):
        f = tt.zeros_like(M)
        rx = self._Rx(theta)[0]
        for l in range(self.ydeg + 1):
            start = self._nwig(l - 1)
            stop = self._nwig(l)
            Rxl = tt.reshape(rx[start:stop], (2 * l + 1, 2 * l + 1))
            f = tt.set_subtensor(
                f[:, l ** 2 : (l + 1) ** 2],
                tt.dot(M[:, l ** 2 : (l + 1) ** 2], Rxl),
            )
        return f

    def _right_project(self, M, theta, inc):
        """Apply the projection operator on the right.

        Specifically, this method returns the dot product `M . R`,
        where `M` is an input matrix and R` is the Wigner rotation matrix
        that transforms a spherical harmonic coefficient vector in the
        input frame to a vector in the observer's frame.
        """
        # Rotate to the sky frame
        M = self._dotRx(M, -inc)

        # Rotate to the correct phase
        M = self._tensordotRz(M, theta)

        # Rotate to the polar frame
        M = self._dotRx(M, 0.5 * np.pi)

        return M

    def _compute(self, theta, inc):
        rTA1 = tt.tile(self._rTA1, (theta.shape[0], 1))
        self._A = self._right_project(rTA1, theta, inc)
        self._AT = tt.transpose(self._A)
