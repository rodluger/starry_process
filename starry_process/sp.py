from .size import SizeIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
from .contrast import ContrastIntegral
from .flux import FluxIntegral
from .math import cho_factor, cho_solve, cast
from .defaults import defaults
from .ops import CheckBoundsOp
import theano.tensor as tt
from theano.ifelse import ifelse
import numpy as np


__all__ = ["StarryProcess"]


class StarryProcess(object):
    def __init__(
        self,
        r=defaults["r"],
        a=defaults["a"],
        b=defaults["b"],
        c=defaults["c"],
        n=defaults["n"],
        i=defaults["i"],
        p=defaults["p"],
        **kwargs,
    ):
        # Spherical harmonic degree of the process
        self._ydeg = kwargs.get("ydeg", defaults["ydeg"])
        assert self._ydeg > 10, "Degree of map must be > 10."
        self._nylm = (self._ydeg + 1) ** 2

        # Initialize the Ylm integral ops
        self.size = SizeIntegral(r, **kwargs)
        self.latitude = LatitudeIntegral(a, b, child=self.size, **kwargs)
        self.longitude = LongitudeIntegral(child=self.latitude, **kwargs)
        self.contrast = ContrastIntegral(c, n, child=self.longitude, **kwargs)

        # Mean and covariance of the Ylm process
        self._mean_ylm = self.contrast.mean()
        self._cov_ylm = self.contrast.cov()
        self._cho_cov_ylm = cho_factor(self._cov_ylm)
        self._LInv = cho_solve(
            self._cho_cov_ylm, tt.eye((self._ydeg + 1) ** 2)
        )
        self._LInvmu = cho_solve(self._cho_cov_ylm, self._mean_ylm)

        # Initialize the flux integral op
        self.flux = FluxIntegral(i, p, child=self.contrast, **kwargs)

        # Seed the randomizer
        self.random = tt.shared_randomstreams.RandomStreams(
            kwargs.get("seed", 0)
        )

    @property
    def mean_ylm(self):
        return self._mean_ylm

    @property
    def cov_ylm(self):
        return self._cov_ylm

    @property
    def cho_cov_ylm(self):
        return self._cho_cov_ylm

    def sample_ylm(self, nsamples=1):
        u = self.random.normal((self._nylm, nsamples))
        return tt.transpose(
            self.mean_ylm[:, None] + tt.dot(self.cho_cov_ylm, u)
        )

    def sample_ylm_conditional(
        self,
        t,
        flux,
        data_cov,
        baseline_mean=0.0,
        baseline_var=0.0,
        nsamples=1,
    ):
        # Get the full data covariance
        t = cast(t)
        data_cov = cast(data_cov)
        if data_cov.ndim == 0:
            C = data_cov * tt.eye(t.shape[0])
        elif data_cov.ndim == 1:
            C = tt.diag(data_cov)
        else:
            C = data_cov

        # TODO: If we're not marginalizing over the baseline,
        # we don't need to instantiate the full covariance matrix!

        # Marginalize over the baseline; note we're adding
        # `baseline_var` to *every* entry in the covariance matrix
        C += baseline_var

        # Compute C^-1 . A
        A = self.flux._design_matrix(t)
        cho_C = cho_factor(C)
        CInvA = cho_solve(cho_C, A)

        # Compute W = A^T . C^-1 . A + L^-1
        W = tt.dot(tt.transpose(A), CInvA) + self._LInv

        # Compute the conditional mean and covariance
        cho_W = cho_factor(W)
        M = cho_solve(cho_W, tt.transpose(CInvA))
        ymu = tt.dot(M, cast(flux) - baseline_mean) + cho_solve(
            cho_W, self._LInvmu
        )
        ycov = cho_solve(cho_W, tt.eye(cho_W.shape[0]))
        cho_ycov = cho_factor(ycov)

        # Sample from it
        u = self.random.normal((self._nylm, nsamples))
        return tt.transpose(ymu[:, None] + tt.dot(cho_ycov, u))

    def mean(self, t):
        return self.flux.mean(cast(t))

    def cov(self, t):
        return self.flux.cov(cast(t))

    def sample(self, t, nsamples=1):
        ylm = self.sample_ylm(nsamples=nsamples)
        return tt.transpose(
            tt.dot(self.flux._design_matrix(cast(t)), tt.transpose(ylm))
        )

    def log_jac(self):
        """
        The log of the absolute value of the determinant of the Jacobian matrix.

        The spot latitude is Beta-distributed with shape parameters
        `a` and `b`, equal to the log of the traditional `alpha` 
        and `beta` parameters of the Beta distribution, normalized and scaled
        to the range `[0, 1]`. From Bayes' theorem, the joint posterior in 
        these two quantities is

            p(a, b | data) ~ p(data | a, b) * p(a, b)

        However, this is a rather awkward parametrization, since it's hard to
        visualize how exactly `a` and `b` (or `alpha` and `beta`) 
        determine quantities we actually care about, such as the mean `mu` and 
        standard deviation `sigma` of the distribution. This parameterization 
        is especially clumsy when it comes to specifying the prior `p(a, b)`, 
        since any prior on these quantities will imply a very different prior 
        on `mu` and `sigma`. In most cases, we probably want to place a prior 
        on `mu` and `sigma` directly. We can do this by noting that

            p(a, b) = p(mu, sigma) * J

        where

            J = | dmu / da * dsigma / db - dmu / db * dsigma / da |

        is the absolute value of the determinant of the Jacobian matrix.

        Thus, to enforce a uniform prior on `mu` and `sigma`, sample
        in `a` and `b` with a uniform prior in the range `[0, 1]`
        and multiply the PDF by `J`. Since we're in log space, you'll want 
        to add `log J` (the value returned by this function) to the
        log likelihood.

        """
        return self.latitude.log_jac()

    def log_likelihood(
        self, t, flux, data_cov, baseline_mean=0.0, baseline_var=0.0,
    ):
        """
        Compute the log marginal likelihood of a light curve.

        Args:
            t (array): The time array.
            flux (array): The array of observed flux values.
            data_cov (scalar/vector/matrix): The data covariance matrix.

        Returns:
            The log marginal likelihood of the `flux` vector conditioned on
            the the current properties of the model. This is the likelihood 
            marginalized over all possible spherical harmonic vectors.

        """
        # Get the flux gp mean and covariance
        gp_mean = self.mean(t)
        gp_cov = self.cov(t)
        K = gp_mean.shape[0]

        # Get the full data covariance
        data_cov = cast(data_cov)
        if data_cov.ndim == 0:
            C = data_cov * tt.eye(K)
        elif data_cov.ndim == 1:
            C = tt.diag(data_cov)
        else:
            C = data_cov

        # Covariances add!
        gp_cov += C

        # Marginalize over the baseline; note that we are adding
        # `baseline_var` to *every* entry in the covariance matrix
        # To see why, c.f. Equation (4) in Luger et al. (2017),
        # where `A` is a column vector of ones (our baseline regressor)
        # and `Lambda` is our prior baseline variance
        gp_cov += baseline_var

        # Cholesky factorization
        cho_gp_cov = cho_factor(gp_cov)

        # Compute the marginal likelihood
        r = tt.reshape(flux - gp_mean - baseline_mean, (-1, 1))
        lnlike = -0.5 * tt.dot(tt.transpose(r), cho_solve(cho_gp_cov, r))
        lnlike -= tt.sum(tt.log(tt.diag(cho_gp_cov)))
        lnlike -= 0.5 * K * tt.log(2 * np.pi)

        # NANs --> -inf
        return ifelse(tt.isnan(lnlike[0, 0]), -np.inf, lnlike[0, 0])
