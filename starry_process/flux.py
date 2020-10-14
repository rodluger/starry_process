# -*- coding: utf-8 -*-
from .ops import RxOp, tensordotRzOp, rTA1Op, CheckBoundsOp
from .wigner import R
from .defaults import defaults
from .math import cast
from scipy.special import gamma, hyp2f1
import numpy as np
import theano
import theano.tensor as tt
from theano.ifelse import ifelse
from tqdm import tqdm
import os


__all__ = ["FluxIntegral"]


# Get cache path
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class FluxIntegral:
    def __init__(
        self,
        i,
        p,
        child=None,
        ydeg=defaults["ydeg"],
        angle_unit=defaults["angle_unit"],
        clobber=False,
        **kwargs
    ):
        # General
        assert child is not None
        self._child = child
        self._mean_ylm = self._child.mean()
        self._cov_ylm = self._child.cov()
        self._ydeg = ydeg
        self._nylm = (self._ydeg + 1) ** 2
        if angle_unit.startswith("deg"):
            self._angle_fac = np.pi / 180
        elif angle_unit.startswith("rad"):
            self._angle_fac = 1.0
        else:
            raise ValueError("Invalid `angle_unit`.")

        # Set up the ops
        self._tensordotRz = tensordotRzOp(ydeg, **kwargs)
        self._Rx = RxOp(ydeg, **kwargs)
        self._R = R(
            self._ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )
        self._CC = tt.extra_ops.CpuContiguous()

        # Get the moments of the Ylm process in the polar frame
        self._ez = tt.transpose(
            self._dotRx(tt.reshape(self._mean_ylm, (1, -1)), 0.5 * np.pi)
        )
        mom2 = self._CC(
            self._cov_ylm + tt.outer(self._mean_ylm, self._mean_ylm)
        )
        tmp = self._CC(tt.transpose(self._dotRx(mom2, 0.5 * np.pi)))
        self._Ez = self._dotRx(tmp, 0.5 * np.pi)

        # Pre-compute the integrals
        self._precompute(clobber=clobber)

        # Ingest the parameters
        self.update(i, p)

    @property
    def _cache_file(self):
        return os.path.join(CACHE_PATH, "flux_{}.npz".format(self._ydeg))

    def update(self, i, p):
        """
        Update the process with new values of the inclination and period.

        """
        self._p = CheckBoundsOp(name="p", lower=0, upper=np.inf)(p)
        if i is None:

            # We're going to marginalize over inclination
            self._i = None

            # Compute the flux mean marginalized over inclination
            self._mean_marg = tt.sum(
                [
                    tt.dot(self._t[l], self._ez[slice(l ** 2, (l + 1) ** 2)])
                    for l in range(self._ydeg + 1)
                ]
            )

        else:

            # We're conditioning the GP on a value of the inclination
            self._i = CheckBoundsOp(name="i", lower=0, upper=0.5 * np.pi)(
                i * self._angle_fac
            )

            # Update the flux mean at this inclination
            self._mean_cond = tt.dot(
                self._design_matrix(cast([0.0])), self._mean_ylm
            )[0]

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

    def _rotate(self, theta):
        """
        Rotate a moment matrix in phase.
        TODO: This can be sped up, since it's not a tensordot!
        
        """
        return self._tensordotRz(self._Ez, tt.tile(theta, self._nylm))

    def _G(self, j, i):
        """
        This is the integral of
         
            cos(x / 2)^i sin(x / 2)^j sin(x)
        
        from 0 to pi/2.
        """
        return 2 * gamma(1 + 0.5 * i) * gamma(1 + 0.5 * j) / gamma(
            0.5 * (4 + i + j)
        ) - (2 ** (1 - 0.5 * i) / (2 + i)) * hyp2f1(
            1 + 0.5 * i, -0.5 * j, 2 + 0.5 * i, 0.5
        )

    def _precompute(self, clobber=False):
        """
        Pre-compute the inclination marginalization integrals.
        
        """

        if clobber or not os.path.exists(self._cache_file):

            print("Pre-computing the inclination integrals...")

            # The flux integral operator
            self._rTA1 = rTA1Op(self._ydeg)().eval()

            # The marginalization integral
            G = np.array(
                [
                    [self._G(i, j) for i in range(4 * self._ydeg + 1)]
                    for j in range(4 * self._ydeg + 1)
                ]
            )

            # First moment integral
            self._t = [None for l in range(self._ydeg + 1)]
            for l in range(self._ydeg + 1):
                m = np.arange(-l, l + 1)
                i = slice(l ** 2, (l + 1) ** 2)
                self._t[l] = self._rTA1[i] @ self._R[l] @ G[l - m, l + m]

            # Second moment integral
            self._T = np.zeros((self._nylm, self._nylm))
            for i in tqdm(range(self._nylm)):
                l1 = int(np.floor(np.sqrt(i)))
                k = np.arange(l1 ** 2, (l1 + 1) ** 2)
                k0 = np.arange(2 * l1 + 1).reshape(-1, 1)
                for p in range(self._nylm):
                    l2 = int(np.floor(np.sqrt(p)))
                    j = np.arange(l2 ** 2, (l2 + 1) ** 2)
                    j0 = np.arange(2 * l2 + 1).reshape(1, -1)
                    Wik = self._rTA1[i] * self._R[l1][i - l1 ** 2, k - l1 ** 2]
                    Wjp = self._rTA1[j] @ self._R[l2][j - l2 ** 2, p - l2 ** 2]
                    M = G[k0 + j0, 2 * l1 - k0 + 2 * l2 - j0]
                    self._T[k, p] += Wik @ M @ Wjp

            # Save
            tkwargs = {
                "t{:d}".format(l): self._t[l] for l in range(self._ydeg + 1)
            }
            np.savez(self._cache_file, rTA1=self._rTA1, T=self._T, **tkwargs)

            print("Done.")

        else:

            data = np.load(self._cache_file)
            self._rTA1 = data["rTA1"]
            self._T = data["T"]
            self._t = [data["t{:d}".format(l)] for l in range(self._ydeg + 1)]

    def _design_matrix(self, t):
        theta = 2 * np.pi / self._p * cast(t, vectorize=True)
        rTA1 = tt.tile(self._rTA1, (theta.shape[0], 1))
        return self._right_project(rTA1, theta, self._i)

    def mean(self, t):
        if self._i is None:
            # Marginalized over inclination
            return self._mean_marg * tt.ones_like(cast(t, vectorize=True))
        else:
            # Conditioned on inclination
            return self._mean_cond * tt.ones_like(cast(t, vectorize=True))

    def cov(self, t, npts=None):
        if self._i is None:

            # Marginalized over inclination

            # This is slower to compute, so we compute the kernel on
            # a 1d grid of theta differences, then interpolate to the
            # full covariance matrix.

            # First, compute the variance (in case len(t) == 1)
            t = cast(t, vectorize=True)
            E0 = self._rotate(2 * np.pi / self._p * t[0])
            var = (tt.tensordot(self._T, E0) - self._mean_marg ** 2) * tt.eye(
                1
            )

            # Number of interpolation points. For a regular grid,
            # the default returns the exact covariance.
            if npts is None:
                npts = tt.maximum(1, t.shape[0] - 1)

            # Evaluate the kernel on a regular 1d grid in theta
            dx = 2 * np.pi / npts
            xp = tt.arange(-dx, 2 * np.pi + 2.5 * dx, dx)
            E = tt.transpose(
                theano.scan(
                    fn=self._rotate, outputs_info=None, sequences=[xp],
                )[0]
            )
            mom2 = tt.tensordot(self._T, E)
            yp = mom2 - self._mean_marg ** 2

            # We need to know the value of the kernel at the following points:
            theta = 2 * np.pi / self._p * t
            x = tt.reshape(tt.abs_(theta[:, None] - theta[None, :]), (-1,))

            # Compute the interpolant
            y0 = yp[:-3]
            y1 = yp[1:-2]
            y2 = yp[2:-1]
            y3 = yp[3:]
            a0 = y1
            a1 = -y0 / 3.0 - 0.5 * y1 + y2 - y3 / 6.0
            a2 = 0.5 * (y0 + y2) - y1
            a3 = 0.5 * ((y1 - y2) + (y3 - y0) / 3.0)
            inds = tt.cast(tt.floor(x / dx), "int64")
            x0 = (x - xp[inds + 1]) / dx
            cov = (
                a0[inds]
                + a1[inds] * x0
                + a2[inds] * x0 ** 2
                + a3[inds] * x0 ** 3
            )

            # Reshape to 2D and we're done
            cov = tt.reshape(cov, (t.shape[0], t.shape[0]))

            # On the off chance that len(t) == 1, return the variance instead
            return ifelse(tt.eq(t.shape[0], 1), var, cov)

        else:
            # Conditioned on inclination
            A = self._design_matrix(t)
            return tt.dot(tt.dot(A, self._cov_ylm), tt.transpose(A))
