# -*- coding: utf-8 -*-
import theano.tensor as tt

__all__ = ["FluxDesignMatrix"]


class FluxDesignMatrix(object):
    def __init__(self):
        raise NotImplementedError("TODO!")

    def right_project(self, M, theta, inc):
        """Apply the projection operator on the right.

        Specifically, this method returns the dot product `M . R`,
        where `M` is an input matrix and R` is the Wigner rotation matrix
        that transforms a spherical harmonic coefficient vector in the
        input frame to a vector in the observer's frame.
        """
        # Trivial case
        if self.ydeg == 0:
            return M

        # Rotate to the sky frame
        M = self.dotRx(M, -inc)

        # Rotate to the correct phase
        M = self.tensordotRz(M, theta)

        # Rotate to the polar frame
        M = self.dotRx(M, 0.5 * np.pi)

        return M

    def compute(self, theta, inc):
        rTA1 = tt.tile(self.rTA1, (theta.shape[0], 1))
        X = self.right_project(rTA1, inc, theta)
        return X
