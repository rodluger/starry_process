from .size import SizeIntegral
from .contrast import ContrastIntegral
from .latitude import LatitudeIntegral
from .longitude import LongitudeIntegral
from .flux import FluxDesignMatrix
from .math import cho_factor, cho_solve, cast
import theano.tensor as tt
from theano.ifelse import ifelse
import numpy as np


class StarryProcess(object):
    def __init__(self, ydeg=15, **kwargs):
        assert ydeg > 10, "Degree of map must be > 10."
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2

        self.contrast = ContrastIntegral(self, **kwargs)
        self.longitude = LongitudeIntegral(self.contrast, **kwargs)
        self.latitude = LatitudeIntegral(self.longitude, **kwargs)
        self.size = SizeIntegral(self.latitude, **kwargs)
        self.design = FluxDesignMatrix(self.ydeg)

        self._mean_ylm = None
        self._cov_ylm = None
        self._cho_cov_ylm = None

        # NB: Change this by setting `self.random.seed(XXX)`
        self.random = tt.shared_randomstreams.RandomStreams(0)

    def mean_ylm(self):
        if (self._mean_ylm is None) or (self.contrast.e is None):
            self._mean_ylm = self.contrast.first_moment()
        return self._mean_ylm

    def cov_ylm(self):
        if (self._cov_ylm is None) or (self.contrast.eigE is None):
            e4 = self.contrast.first_moment()
            eigE4 = self.contrast.second_moment()
            self._cov_ylm = tt.dot(eigE4, tt.transpose(eigE4)) - tt.outer(
                e4, e4
            )
            self._cho_cov_ylm = None
        return self._cov_ylm

    def sample_ylm(self, nsamples=1, eps=1e-12):
        if (
            (self.contrast.eigE is None)
            or (self._cov_ylm is None)
            or (self._cho_cov_ylm is None)
        ):
            self._cho_cov_ylm = cho_factor(
                self.cov_ylm() + tt.eye(self.N) * eps
            )
        u = self.random.normal((self.N, nsamples))
        return tt.transpose(
            self.mean_ylm()[:, None] + tt.dot(self._cho_cov_ylm, u)
        )

    def mean(self, t):
        return tt.dot(self.design(t), self.mean_ylm())

    def cov(self, t):
        A = self.design(t)
        return tt.dot(tt.dot(A, self.cov_ylm()), tt.transpose(A))

    def sample(self, t, nsamples=1, eps=1e-12):
        ylm = self.sample_ylm(nsamples=nsamples, eps=eps)
        return tt.dot(ylm, tt.transpose(self.design(t)))

    def lnlike(self, t, flux, data_cov, baseline_mean=0.0, baseline_var=0.0):
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
        return lnlike[0, 0]
