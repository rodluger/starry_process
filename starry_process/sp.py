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

    def log_likelihood(
        self, t, flux, data_cov, baseline_mean=0.0, baseline_var=0.0
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
        gp_cov = C + self.cov(t)

        # Marginalize over the baseline
        gp_cov += baseline_var

        # Cholesky factorization
        cho_gp_cov = cho_factor(gp_cov)

        # Compute the marginal likelihood
        K = t.shape[0]
        r = tt.reshape(flux - self.mean(t) - baseline_mean, (-1, 1))
        lnlike = -0.5 * tt.dot(tt.transpose(r), cho_solve(cho_gp_cov, r))
        lnlike -= tt.sum(tt.log(tt.diag(cho_gp_cov)))
        lnlike -= 0.5 * K * tt.log(2 * np.pi)

        # NANs --> -inf
        return ifelse(tt.isnan(lnlike[0, 0]), -np.inf, lnlike[0, 0])
