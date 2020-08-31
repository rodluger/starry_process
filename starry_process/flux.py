# -*- coding: utf-8 -*-
from .math import theano_math, numpy_math

__all__ = ["FluxDesignMatrix"]


class FluxDesignMatrix(object):
    def __init__(self, ydeg, **kwargs):
        self._math = (
            theano_math if kwargs.get("use_theano", True) else numpy_math
        )
        self.ydeg = ydeg
        self._set = False
        try:
            self._Rx = self._math.ops.RxOp(ydeg, **kwargs)
            self._tensordotRz = self._math.ops.tensordotRzOp(ydeg, **kwargs)
            self._rTA1 = self._math.ops.rTA1Op(ydeg, **kwargs)()
        except AttributeError:
            self._Rx = None
            self._tensordotRz = None
            self._rTA1 = None

    def set_params(self, period, inc):
        self._omega = 2 * self._math.pi / period
        self._inc = inc * self._math.pi / 180
        self._set = True

    def _nwig(self, l):
        return ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3

    def _dotRx(self, M, theta):
        f = self._math.zeros_like(M)
        rx = self._Rx(theta)[0]
        for l in range(self.ydeg + 1):
            start = self._nwig(l - 1)
            stop = self._nwig(l)
            Rxl = self._math.reshape(rx[start:stop], (2 * l + 1, 2 * l + 1))
            try:
                f = self._math.set_subtensor(
                    f[:, l ** 2 : (l + 1) ** 2],
                    self._math.dot(M[:, l ** 2 : (l + 1) ** 2], Rxl),
                )
            except AttributeError:
                f[:, l ** 2 : (l + 1) ** 2] = self._math.dot(
                    M[:, l ** 2 : (l + 1) ** 2], Rxl
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
        M = self._dotRx(M, 0.5 * self._math.pi)

        return M

    def __call__(self, t):
        if self._rTA1 is None:
            raise NotImplementedError(
                "Pure Python version of the flux ops not yet implemented."
            )
        assert self._set, "must call `set_params` first."
        theta = self._omega * t
        rTA1 = self._math.tile(self._rTA1, (theta.shape[0], 1))
        return self._right_project(rTA1, theta, self._inc)
