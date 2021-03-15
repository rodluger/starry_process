# -*- coding: utf-8 -*-
from .ops import (
    RxOp,
    tensordotRzOp,
    special_tensordotRzOp,
    rTA1Op,
    rTA1LOp,
    CheckBoundsOp,
)
from .wigner import R
from .defaults import defaults
from .math import cast
from .compat import theano, tt, ifelse
from scipy.special import gamma, hyp2f1
import numpy as np
from tqdm import tqdm
import os


__all__ = ["FluxIntegral"]


class FluxIntegral:
    def __init__(
        self,
        mean_ylm,
        cov_ylm,
        udeg=defaults["udeg"],
        marginalize_over_inclination=defaults["marginalize_over_inclination"],
        covpts=defaults["covpts"],
        ydeg=defaults["ydeg"],
        **kwargs
    ):

        # General
        self._udeg = udeg
        self._marginalize_over_inclination = marginalize_over_inclination
        self._covpts = covpts
        self._mean_ylm = mean_ylm
        self._cov_ylm = cov_ylm
        self._ydeg = ydeg
        self._nylm = (self._ydeg + 1) ** 2
        self._angle_fac = np.pi / 180

        # Set up the ops
        self._special_tensordotRz = special_tensordotRzOp(ydeg, **kwargs)
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

        # Pre-compute stuff that doesn't depend on user inputs
        self._precompute()

        # These parameters are star-specific and are set in
        # calls to `mean()` and `cov()`.
        self._t = None
        self._i = None
        self._p = None
        self._u = None

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

    def _precompute(self):
        """
        Pre-compute some stuff that doesn't depend on
        user inputs.

        """
        # First, we can pre-compute a bunch of stuff
        # using `numpy`, as it doesn't depend on tensor
        # variables.

        # The marginalization integral
        G = np.array(
            [
                [self._G(i, j) for i in range(4 * self._ydeg + 1)]
                for j in range(4 * self._ydeg + 1)
            ]
        )

        # First moment integral
        self._wnp = [None for l in range(self._ydeg + 1)]
        for l in range(self._ydeg + 1):
            m = np.arange(-l, l + 1)
            i = slice(l ** 2, (l + 1) ** 2)
            self._wnp[l] = self._R[l] @ G[l - m, l + m]

        # Second moment integral. Apologies for
        # how opaque this all is, since it's the result of
        # a few hours of tinkering with linear algebra to
        # maximize the number of operations *before*
        # we get theano involved.
        Q = np.empty(
            (
                2 * self._ydeg + 1,
                2 * self._ydeg + 1,
                2 * self._ydeg + 1,
                self._nylm,
            )
        )
        for l1 in range(self._ydeg + 1):
            k = np.arange(l1 ** 2, (l1 + 1) ** 2)
            k0 = np.arange(2 * l1 + 1).reshape(-1, 1)
            for p in range(self._nylm):
                l2 = int(np.floor(np.sqrt(p)))
                j = np.arange(l2 ** 2, (l2 + 1) ** 2)
                j0 = np.arange(2 * l2 + 1).reshape(1, -1)
                L = (
                    self._R[l1][l1, k - l1 ** 2]
                    @ G[k0 + j0, 2 * l1 - k0 + 2 * l2 - j0]
                )
                R = self._R[l2][j - l2 ** 2, p - l2 ** 2].T
                Q[l1, : 2 * l1 + 1, : 2 * l2 + 1, p] = L @ R
        self._Wnp = np.empty((self._nylm, self._nylm))
        for l1 in range(self._ydeg + 1):
            i = np.arange(l1 ** 2, (l1 + 1) ** 2)
            for l2 in range(self._ydeg + 1):
                j = np.arange(l2 ** 2, (l2 + 1) ** 2)
                self._Wnp[i.reshape(-1, 1), j.reshape(1, -1)] = Q[
                    l1, : 2 * l1 + 1, l2, j
                ].T

    def _compute_inclination_integrals(self):

        # In the computation of the second moment below, we implicitly
        # make use of the fact that `rTA1 = 0` if `m != 0` because
        # of symmetry. This vastly reduces the number of operations
        # we need to do!

        if self._udeg > 0:

            # Flux integral op
            self._rTA1 = rTA1LOp(ydeg=self._ydeg, udeg=self._udeg)(self._u)

            # First moment
            self._w = [None for l in range(self._ydeg + 1)]
            for l in range(self._ydeg + 1):
                i = slice(l ** 2, (l + 1) ** 2)
                self._w[l] = tt.dot(self._rTA1[i], self._wnp[l])

            # Second moment
            m0 = np.array([l ** 2 + l for l in range(self._ydeg + 1)])
            Z = tt.outer(self._rTA1[m0], self._rTA1[m0])
            self._W = tt.zeros((self._nylm, self._nylm))
            for l1 in range(self._ydeg + 1):
                i = np.arange(l1 ** 2, (l1 + 1) ** 2).reshape(-1, 1)
                for l2 in range(self._ydeg + 1):
                    j = np.arange(l2 ** 2, (l2 + 1) ** 2).reshape(1, -1)
                    self._W = tt.set_subtensor(
                        self._W[i, j], self._Wnp[i, j] * Z[l1, l2]
                    )

        else:

            # Get the numeric value of the flux op, since it
            # doesn't depend on any user inputs
            self._rTA1 = rTA1Op(ydeg=self._ydeg)().eval()

            # First moment
            self._w = [None for l in range(self._ydeg + 1)]
            for l in range(self._ydeg + 1):
                i = slice(l ** 2, (l + 1) ** 2)
                self._w[l] = self._rTA1[i] @ self._wnp[l]

            # Second moment
            m0 = np.array([l ** 2 + l for l in range(self._ydeg + 1)])
            Z = np.outer(self._rTA1[m0], self._rTA1[m0])
            self._W = np.zeros((self._nylm, self._nylm))
            for l1 in range(self._ydeg + 1):
                i = np.arange(l1 ** 2, (l1 + 1) ** 2).reshape(-1, 1)
                for l2 in range(self._ydeg + 1):
                    j = np.arange(l2 ** 2, (l2 + 1) ** 2).reshape(1, -1)
                    self._W[i, j] = self._Wnp[i, j] * Z[l1, l2]

    def _set_params(self, t, i, p, u):
        # Ingest
        t = cast(t, vectorize=True)
        i = CheckBoundsOp(name="i", lower=0, upper=0.5 * np.pi)(
            i * self._angle_fac
        )
        p = CheckBoundsOp(name="p", lower=0, upper=np.inf)(p)
        u = cast(u, vectorize=True)[: self._udeg]

        # Check if they are different from the cached variables
        if (
            (self._t != t)
            or (self._i != i)
            or (self._p != p)
            or (self._u != u)
        ):
            # We need to re-compute everything
            self._t = t
            self._i = i
            self._p = p
            self._u = u
            self._compute()

    def _interpolate_cov(self):
        """
        Interpolate the pre-computed kernel onto a
        grid of time lags in 2D to get the full covariance.

        """
        theta = 2 * np.pi * tt.mod(self._t / self._p, 1.0)
        x = tt.reshape(tt.abs_(theta[:, None] - theta[None, :]), (-1,))
        inds = tt.cast(tt.floor(x / self._dx), "int64")
        x0 = (x - self._xp[inds + 1]) / self._dx
        cov = tt.reshape(
            self._a0[inds]
            + self._a1[inds] * x0
            + self._a2[inds] * x0 ** 2
            + self._a3[inds] * x0 ** 3,
            (theta.shape[0], theta.shape[0]),
        )

        # If len(theta) == 1, return the *variance* instead
        cov = ifelse(tt.eq(theta.shape[0], 1), self._var, cov)
        return cov

    def _design_matrix(self):
        theta = 2 * np.pi * tt.mod(self._t / self._p, 1.0)
        rTA1 = tt.tile(self._rTA1, (theta.shape[0], 1))
        return self._right_project(rTA1, theta, self._i)

    def _compute(self):
        """
        Compute some vectors and matrices used in the
        evaluation of the mean and covariance.

        """
        # Compute some helper matrices
        self._compute_inclination_integrals()

        # If we're marginalizing over inclination, pre-compute the
        # radial kernel on a fine grid, then interpolate onto the
        # provided time array.
        if self._marginalize_over_inclination:

            # Compute the scalar mean
            self._mean = tt.sum(
                [
                    tt.dot(self._w[l], self._ez[slice(l ** 2, (l + 1) ** 2)])
                    for l in range(self._ydeg + 1)
                ]
            )

            # Compute the *variance*
            self._var = (
                tt.tensordot(self._W, self._Ez) - self._mean ** 2
            ) * tt.eye(1)

            # Evaluate the kernel on a regular 1d grid in theta
            self._dx = 2 * np.pi / self._covpts
            self._xp = tt.arange(
                -self._dx, 2 * np.pi + 2.5 * self._dx, self._dx
            )

            # Compute the batched tensor dot product T_ij R_ilk M_lj
            mom2 = self._special_tensordotRz(self._W, self._Ez, self._xp)

            # The actual covariance
            yp = mom2 - self._mean ** 2

            # Compute the interpolant
            y0 = yp[:-3]
            y1 = yp[1:-2]
            y2 = yp[2:-1]
            y3 = yp[3:]
            self._a0 = y1
            self._a1 = -y0 / 3.0 - 0.5 * y1 + y2 - y3 / 6.0
            self._a2 = 0.5 * (y0 + y2) - y1
            self._a3 = 0.5 * ((y1 - y2) + (y3 - y0) / 3.0)

            # Compute the covariance
            self._cov = self._interpolate_cov()

        else:

            A = self._design_matrix()

            # Compute the scalar mean
            self._mean = tt.dot(A, self._mean_ylm)[0]

            # Compute the covariance
            self._cov = tt.dot(tt.dot(A, self._cov_ylm), tt.transpose(A))

    def design_matrix(self, t, i, p, u):
        """

        """
        self._set_params(t, i, p, u)
        return self._design_matrix()

    def kernel(self, t, i, p, u):
        """

        """
        self._set_params(t, i, p, u)
        if self._marginalize_over_inclination:
            mom2 = self._special_tensordotRz(
                self._W,
                self._Ez,
                2 * np.pi * tt.mod(tt.reshape(self._t, (-1,)) / self._p, 1.0),
            )
            return mom2 - self._mean ** 2
        else:
            return self._cov[0]

    def mean(self, t, i, p, u):
        """

        """
        self._set_params(t, i, p, u)
        return self._mean * tt.ones_like(self._t)

    def cov(self, t, i, p, u):
        """

        """
        self._set_params(t, i, p, u)
        return self._cov
