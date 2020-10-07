from .size import SizeIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
from .design import FluxDesignMatrix
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
        size=defaults["size"],
        latitude=defaults["latitude"],
        longitude=defaults["longitude"],
        contrast=defaults["contrast"],
        **kwargs
    ):
        # Spherical harmonic degree of the process
        self._ydeg = kwargs.get("ydeg", defaults["ydeg"])
        assert self._ydeg > 10, "Degree of map must be > 10."
        self._nylm = (self._ydeg + 1) ** 2

        # Initialize the integral ops
        self.size = SizeIntegral(size, **kwargs)
        self.latitude = LatitudeIntegral(latitude, child=self.size, **kwargs)
        self.longitude = LongitudeIntegral(
            longitude, child=self.latitude, **kwargs
        )
        self.design = FluxDesignMatrix(**kwargs)

        # Contrast
        assert (
            hasattr(contrast, "__len__") and len(contrast) == 2
        ), "Parameter `contrast` must be a tuple containing two values."
        self._c = CheckBoundsOp(name="c", lower=0, upper=1)(contrast[0])
        self._N = CheckBoundsOp(name="N", lower=1, upper=np.inf)(contrast[1])

        # Stability hacks
        eps1 = kwargs.pop("eps1", 1e-12)
        eps2 = kwargs.pop("eps2", 1e-9)
        lam = np.ones(self._nylm) * eps1
        lam[self._ydeg ** 2 :] = eps2
        lam = tt.diag(lam)

        # Pre-compute the moments
        mom1 = self.longitude.first_moment()
        eig_mom2 = self.longitude.second_moment()
        mom2 = tt.dot(eig_mom2, tt.transpose(eig_mom2))
        self.mean_ylm = np.pi * self._c * self._N * mom1
        self.cov_ylm = (
            (np.pi * self._c) ** 2 * self._N * (mom2 - tt.outer(mom1, mom1))
        )
        self.cov_ylm += lam
        self.cho_cov_ylm = cho_factor(self.cov_ylm)
        self._q0 = cho_solve(self.cho_cov_ylm, self.mean_ylm)

        # Seed the randomizer
        self.random = tt.shared_randomstreams.RandomStreams(
            kwargs.get("seed", 0)
        )

    def sample_ylm(self, nsamples=1):
        u = self.random.normal((self._nylm, nsamples))
        return tt.transpose(
            self.mean_ylm[:, None] + tt.dot(self.cho_cov_ylm, u)
        )

    def mean(self, t, period=defaults["period"], inc=defaults["inc"]):
        return tt.dot(self.design(t, period, inc), self.mean_ylm)

    def cov(self, t, period=defaults["period"], inc=defaults["inc"]):
        A = self.design(t, period, inc)
        return tt.dot(tt.dot(A, self.cov_ylm), tt.transpose(A))

    def sample(
        self, t, period=defaults["period"], inc=defaults["inc"], nsamples=1
    ):
        ylm = self.sample_ylm(nsamples=nsamples)
        return tt.transpose(
            tt.dot(self.design(t, period, inc), tt.transpose(ylm))
        )

    def sample_ylm_conditional(
        self,
        t,
        flux,
        data_cov,
        period=defaults["period"],
        inc=defaults["inc"],
        baseline_mean=0.0,
        baseline_var=0.0,
        nsamples=1,
    ):
        # Get the full data covariance
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
        A = self.design(t, period, inc)
        cho_C = cho_factor(C)
        CInvA = cho_solve(cho_C, A)

        # Compute W = A^T . C^-1 . A + L^-1
        W = tt.dot(tt.transpose(A), CInvA)
        W += cho_solve(self.cho_cov_ylm, tt.eye((self._ydeg + 1) ** 2))
        LInvmu = cho_solve(self.cho_cov_ylm, self.mean_ylm)

        # Compute the conditional mean and covariance
        cho_W = cho_factor(W)
        M = cho_solve(cho_W, tt.transpose(CInvA))
        ymu = tt.dot(M, cast(flux) - baseline_mean) + cho_solve(cho_W, LInvmu)
        ycov = cho_solve(cho_W, tt.eye(cho_W.shape[0]))
        cho_ycov = cho_factor(ycov)

        # Sample from it
        u = self.random.normal((self._nylm, nsamples))
        return tt.transpose(ymu[:, None] + tt.dot(cho_ycov, u))

    def log_jac(self):
        """
        The log of the absolute value of the determinant of the Jacobian matrix.

        This term should be added to the log likelihood when one wishes to
        sample in the `a` and `b` parameters of the spot size and latitude 
        distributions, while placing a prior on the (more interpretable)
        quantities `mu` and `sigma`.

        The spot size and latitude are Beta-distributed, with shape parameters
        `a` and `b`, equal to the log of the traditional `alpha` 
        and `beta` parameters of the Beta distribution, normalized and scaled
        to the range `[0, 1]`. From Bayes' theorem, the joint posterior in 
        these two quantities is

            p(a, b | data) ~ p(data | a, b) * p(a, b)

        However, this is a rather awkward parametrization, since it's hard to
        visualize how exactly `a` and `b` (or `alpha` and `beta`) 
        determine quantities we actually care about, such as the mean `mu` and 
        standard deviation `sigma` of the distributions. This parameterization 
        is especially clumsy when it comes to specifying the prior `p(a, b)`, 
        since any prior on these quantities will imply a very different prior 
        on `mu` and `sigma`. In most cases, we probably want to place a prior 
        on `mu` and `sigma` directly. We can do this by noting that

            p(a, b) = p(mu, sigma) * J

        where

            J = | dmu / da * dsigma / db - dmu / db * dsigma / da |

        is the absolute value of the determinant of the Jacobian matrix.

        Thus, to enforce a uniform prior on `mu` and `sigma`, sample
        in `a` and `b` with a uniform prior in the range `[0, 1`]
        and multiply the PDF by `J`. Since we're in log space, you'll want 
        to add `log J` (the value returned by this function) to the
        log likelihood.

        """
        return self.size.log_jac() + self.latitude.log_jac()

    def log_likelihood(
        self,
        t,
        flux,
        data_cov,
        period=defaults["period"],
        inc=defaults["inc"],
        baseline_mean=0.0,
        baseline_var=0.0,
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
        A = self.design(t, period, inc)
        gp_mean = tt.dot(A, self.mean_ylm)
        gp_cov = tt.dot(tt.dot(A, self.cov_ylm), tt.transpose(A))

        # Get the full data covariance
        data_cov = cast(data_cov)
        if data_cov.ndim == 0:
            C = data_cov * tt.eye(t.shape[0])
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
        K = t.shape[0]
        r = tt.reshape(flux - gp_mean - baseline_mean, (-1, 1))
        lnlike = -0.5 * tt.dot(tt.transpose(r), cho_solve(cho_gp_cov, r))
        lnlike -= tt.sum(tt.log(tt.diag(cho_gp_cov)))
        lnlike -= 0.5 * K * tt.log(2 * np.pi)

        # NANs --> -inf
        return ifelse(tt.isnan(lnlike[0, 0]), -np.inf, lnlike[0, 0])
