from .size import SizeIntegral
from .contrast import ContrastIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
from .flux import FluxDesignMatrix
from .math import cho_factor, cho_solve, cast
import theano.tensor as tt
from theano.ifelse import ifelse
import numpy as np


__all__ = ["StarryProcess"]


class StarryProcess(object):
    def __init__(self, ydeg=15, **kwargs):
        assert ydeg > 10, "Degree of map must be > 10."
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2

        # Initialize the integral ops
        self.size = SizeIntegral(self.ydeg, **kwargs)
        self.latitude = LatitudeIntegral(self.ydeg, child=self.size, **kwargs)
        self.longitude = LongitudeIntegral(
            self.ydeg, child=self.latitude, **kwargs
        )
        self.contrast = ContrastIntegral(
            self.ydeg, child=self.longitude, **kwargs
        )
        self.design = FluxDesignMatrix(self.ydeg, **kwargs)

        # NB: Change this by setting `self.random.seed(XXX)`
        self.random = tt.shared_randomstreams.RandomStreams(0)

    def mean_ylm(self):
        return self.contrast.first_moment()

    def cov_ylm(self):
        e4 = self.contrast.first_moment()
        eigE4 = self.contrast.second_moment()
        return tt.dot(eigE4, tt.transpose(eigE4)) - tt.outer(e4, e4)

    def sample_ylm(self, nsamples=1, eps=1e-12):
        cho_cov_ylm = cho_factor(self.cov_ylm() + tt.eye(self.N) * eps)
        u = self.random.normal((self.N, nsamples))
        return tt.transpose(self.mean_ylm()[:, None] + tt.dot(cho_cov_ylm, u))

    def mean(self, t):
        return tt.dot(self.design(t), self.mean_ylm())

    def cov(self, t):
        A = self.design(t)
        return tt.dot(tt.dot(A, self.cov_ylm()), tt.transpose(A))

    def sample(self, t, nsamples=1, eps=1e-12):
        ylm = self.sample_ylm(nsamples=nsamples, eps=eps)
        return tt.dot(ylm, tt.transpose(self.design(t)))

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
        return (
            self.size._log_jac()
            + self.latitude._log_jac()
            + self.contrast._log_jac()
        )

    def log_likelihood(
        self, t, flux, data_cov, N=1.0, baseline_mean=0.0, baseline_var=0.0
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
        # Get the full data covariance
        data_cov = cast(data_cov)
        if data_cov.ndim == 0:
            C = data_cov * tt.eye(t.shape[0])
        elif data_cov.ndim == 1:
            C = tt.diag(data_cov)
        else:
            C = data_cov

        # GP covariance from e.g., Luger et al. (2017)
        gp_cov = C + N ** 2 * self.cov(t)

        # Marginalize over the baseline
        gp_cov += baseline_var

        # Cholesky factorization
        cho_gp_cov = cho_factor(gp_cov)

        # Compute the marginal likelihood
        K = t.shape[0]
        r = tt.reshape(flux - N * self.mean(t) - baseline_mean, (-1, 1))
        lnlike = -0.5 * tt.dot(tt.transpose(r), cho_solve(cho_gp_cov, r))
        lnlike -= tt.sum(tt.log(tt.diag(cho_gp_cov)))
        lnlike -= 0.5 * K * tt.log(2 * np.pi)

        # NANs --> -inf
        return ifelse(tt.isnan(lnlike[0, 0]), -np.inf, lnlike[0, 0])
