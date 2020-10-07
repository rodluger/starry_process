# -*- coding: utf-8 -*-
from .ops import RxOp, tensordotRzOp, rTA1Op, CheckBoundsOp
from .defaults import defaults
from .math import cast
import numpy as np
import theano.tensor as tt


__all__ = ["FluxDesignMatrix"]


class FluxDesignMatrix(object):
    def __init__(
        self,
        ydeg=defaults["ydeg"],
        angle_unit=defaults["angle_unit"],
        **kwargs
    ):
        self._ydeg = ydeg
        if angle_unit.startswith("deg"):
            self._angle_fac = np.pi / 180
        elif angle_unit.startswith("rad"):
            self._angle_fac = 1.0
        else:
            raise ValueError("Invalid `angle_unit`.")
        self._Rx = RxOp(ydeg, **kwargs)
        self._tensordotRz = tensordotRzOp(ydeg, **kwargs)
        self._rTA1 = rTA1Op(ydeg, **kwargs)()

    def _dotRx(self, M, theta):
        f = tt.zeros_like(M)
        rx = self._Rx(theta)[0]
        nwig = lambda l: ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3
        for l in range(self._ydeg + 1):
            start = nwig(l - 1)
            stop = nwig(l)
            Rxl = tt.reshape(rx[start:stop], (2 * l + 1, 2 * l + 1))
            f = tt.set_subtensor(
                f[:, l ** 2 : (l + 1) ** 2],
                tt.dot(M[:, l ** 2 : (l + 1) ** 2], Rxl),
            )
        return f

    def _right_project(self, M, theta, inc):
        """Apply the projection operator on the right.

        Specifically, this method returns the dot product `M . R`,
        where `M` is an input matrix and `R` is the Wigner rotation matrix
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

    def __call__(self, t, period, inc):
        self._params = [
            CheckBoundsOp(name="period", lower=0, upper=np.inf)(period),
            CheckBoundsOp(name="inc", lower=0, upper=0.5 * np.pi)(
                inc * self._angle_fac
            ),
        ]
        theta = 2 * np.pi / self._params[0] * t
        rTA1 = tt.tile(self._rTA1, (theta.shape[0], 1))
        return self._right_project(rTA1, theta, self._params[1])
