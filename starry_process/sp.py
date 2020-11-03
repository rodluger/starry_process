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
        normalized=defaults["normalized"],
        marginalize_over_inclination=defaults["marginalize_over_inclination"],
        covpts=defaults["covpts"],
        **kwargs,
    ):
        """
        An interpretable Gaussian process for stellar light curves.

        Args:
            r (scalar, optional): The mean star spot radius in degrees.
                Default is %%defaults["r"]%%. Care should be taken when
                modeling very small spots, as the spherical
                harmonic expansion can typically only model features
                with radius on the order of ``180 / ydeg`` or larger.
                For the default spherical harmonic degree, the minimum
                radius is about 10 degrees. Values below this will
                typically lead to poor approximations, although the
                process will in general still be valid and numerically
                stable.
            a (scalar, optional): Shape parameter of the spot latitude
                distribution. This is equal to the log of the ``alpha``
                parameter characterizing the Beta distribution in the
                cosine of the latitude, scaled to the range ``[0, 1]``.
                Default is %%defaults["a"]%%.
            b (scalar, optional): Shape parameter of the spot latitude
                distribution. This is equal to the log of the ``beta``
                parameter characterizing the Beta distribution in the
                cosine of the latitude, scaled to the range ``[0, 1]``.
                Default is %%defaults["b"]%%.
            c (scalar, optional): The mean spot contrast as a fraction of
                the photospheric intensity. Default is %%defaults["c"]%%.
            n (scalar, optional): The total number of spots. Note that since a
                ``StarryProcess`` does not model spots as discrete features,
                this parameter will not generally have the expected behavior
                when sampling from the prior. In other words, it is unlikely
                that a draw with ``n=10`` will have ten distinct spots when
                visualizing the corresponding stellar surface, nor will it
                necessarily have more spots than a draw with (say) ``n=5``.
                However, this parameter *does* behave correctly in an
                inference setting: the posterior over ``n`` when doing 
                inference on an ensemble of light curves is meaningful and
                should have a mean (on average) equal to the true number of
                spots (assuming all other model assumptions are valid).
                Default is %%defaults["n"]%%.
            normalized (bool, optional): Whether or not the flux observations 
                (passed in calls to ``log_likelihood`` and 
                ``sample_ylm_conditional``) are normalized. Usually, the
                true baseline in stellar photometry is unknown, as it
                requires knowledge of how bright the star would be in the
                absence of star spots. If the baseline is unknown
                (which is almost certainly the case), set this keyword
                to ``True`` and make sure observations are mean- (or median-)
                normalized. Setting this to ``False`` is not recommended for
                usage on real data. Default is %%defaults["normalized"]%%.
            marginalize_over_inclination (bool, optional): Whether or not to
                marginalize over the inclination under the assumption of an
                isotropic prior. Recommended if there are no constraints on
                the inclination of the object. If this is set to ``True``,
                the value of the ``i`` keyword to several of the methods in 
                this class will be ignored. Default is 
                %%defaults["marginalize_over_inclination"]%%.
            covpts (int, optional): The number of grid points on which to
                compute the kernel when ``marginalize_over_inclination`` is
                set to ``True``. Since the full covariance is expensive to
                compute, ``StarryProcess`` instead computes it on a uniform
                grid in phase lag and performs a cubic spline interpolation
                to obtain the covariance matrix. Increasing this number will
                improve the accuracy of the computation at the expense of
                greater runtime. Default is %%defaults["covpts"]%%.
            ydeg (int, optional): The spherical harmonic  degree of the 
                process. Default is %%defaults["ydeg"]%%. Decreasing this
                value will speed up computations but decrease the ability
                to model small features on the surface. Increasing this
                above the default value is not recommended, as it can lead
                to numerical instabilities.

        The following under-the-hood keyword arguments are also accepted:

        Parameters:
            log_alpha_max (float, optional): The maximum value of 
                ``log(alpha)``. Default is %%defaults["log_alpha_max"]%%.
            log_beta_max (float, optional): The maximum value of ``log(beta)``. 
                Default is %%defaults["log_alpha_max"]%%.
            sigma_max (float, optional): The maximum value of the latitude 
                standard deviation in degrees. The latitude distribution 
                becomes extremely non-gaussian for high values of ``sigma``. 
                This value is used to penalize such distributions when 
                computing the jacobian of the transformation.
                Default is %%defaults["sigma_max"]%%.
            compile_args (list, optional): Additional arguments to be passed to
                the C compiler when compiling the ops for this class. Each
                entry in the list should be a tuple of ``(name, value)`` pairs.
                For possible options, see the macros under ``USER CONSTANTS``
                in the header file
                `starry_process/ops/include/constants.h 
                <https://github.com/rodluger/starry_process/blob/master/starry_process/ops/include/constants.h>`_.
            eps1 (float, optional): A small number added to the diagonal of the
                spherical harmonic covariance matrix for stability.
                Default is %%defaults["eps1"]%%.
            eps2 (float, optional): A small number added to terms in the 
                diagonal of the spherical harmonic covariance matrix 
                above degree ``15``, which become particularly unstable.
                Default is %%defaults["eps2"]%%.
            eps3 (float, optional): A small number added to the diagonal of the
                flux covariance matrix when marginalizing over inclination
                for extra stability. Default is %%defaults["eps3"]%%.

        """
        # Spherical harmonic degree of the process
        self._ydeg = kwargs.get("ydeg", defaults["ydeg"])
        assert self._ydeg > 5, "Degree of map must be > 5."
        self._nylm = (self._ydeg + 1) ** 2

        # Is the flux normalized?
        self._normalized = normalized

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
        self._marginalize_over_inclination = marginalize_over_inclination
        self.flux = FluxIntegral(
            child=self.contrast,
            marginalize_over_inclination=marginalize_over_inclination,
            covpts=covpts,
            **kwargs,
        )

        # Seed the randomizer
        self.random = tt.shared_randomstreams.RandomStreams(
            kwargs.get("seed", 0)
        )

    @property
    def mean_ylm(self):
        """
        The mean spherical harmonic coefficient vector.

        """
        return self._mean_ylm

    @property
    def cov_ylm(self):
        """
        The spherical harmonic covariance matrix.
        
        """
        return self._cov_ylm

    @property
    def cho_cov_ylm(self):
        """
        The lower Cholesky factorization of the spherical harmonic covariance.
        
        """
        return self._cho_cov_ylm

    def sample_ylm(self, nsamples=1):
        """
        Draw samples from the prior.

        Args:
            nsamples (int, optional): The number of samples to draw. Default 1.

        """
        u = self.random.normal((self._nylm, nsamples))
        return tt.transpose(
            self.mean_ylm[:, None] + tt.dot(self.cho_cov_ylm, u)
        )

    def sample_ylm_conditional(
        self,
        t,
        flux,
        data_cov,
        i=defaults["i"],
        p=defaults["p"],
        baseline_mean=defaults["baseline_mean"],
        baseline_var=defaults["baseline_var"],
        nsamples=1,
    ):
        """
        Draw samples from the process conditioned on observed flux values.

        Args:
            t (vector): The time array in arbitrary units.
            flux (vector): The array of observed flux values in arbitrary 
                units. In general, the flux should be either mean- or
                median-normalized with zero baseline. If the raw photometry
                is measured in ``counts``, users should compute the ``flux``
                from

                    .. code-block:: python
                    
                        flux = counts / np.median(counts) - 1

                If the baseline is something else (such as unity), users
                may alternatively set the ``baseline_mean`` parameter to
                reflect that.
                Note that if the ``normalized`` keyword passed to this class
                is ``False`` (not recommended for real data), then the flux
                should instead be normalized to the true baseline (i.e., the
                counts one would measure if the star had no spots). 
            data_cov (scalar, vector, or matrix): The data covariance 
                matrix. This may be a scalar equal to the (homoscedastic) 
                variance of the data, a vector equal to the variance of each 
                observation, or a matrix equal to the full covariance of the 
                dataset.
            i (scalar, optional): The inclination of the star in degrees.
                Default is %%defaults["i"]%%. If ``marginalize_over_inclination``
                is set, this argument is ignored.
            p (scalar, optional): The rotational period of the star in the same
                units as ``t``. Default is %%defaults["p"]%%.
            baseline_mean (scalar or vector, optional): The flux baseline to
                subtract when computing the GP likelihood. Default is 
                %%defaults["baseline_mean"]%%.
            baseline_var (scalar or matrix): The variance (square of the
                uncertainty) on the true value of the baseline. This is added
                to every element of the GP covariance matrix in order to
                marginalize over the baseline uncertainty. This may also be a
                matrix specifying the covariance due to additional correlated
                noise unrelated to star spot variability. Default is
                %%defaults["baseline_var"]%%.
            nsamples (int, optional): The number of samples to draw. Default 1.
            
        """
        # TODO
        if self._marginalize_over_inclination:
            raise NotImplementedError(
                "Not yet implemented when marginalizing over inclination."
            )

        # TODO
        if self._normalized:
            raise NotImplementedError(
                "Not yet implemented when the flux is normalized."
            )

        # Get the full data covariance
        flux = cast(flux)
        data_cov = cast(data_cov)
        if data_cov.ndim == 0:
            C = data_cov * tt.eye(flux.shape[0])
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
        A = self.flux._design_matrix(t, i, p)
        cho_C = cho_factor(C)
        CInvA = cho_solve(cho_C, A)

        # Compute W = A^T . C^-1 . A + L^-1
        W = tt.dot(tt.transpose(A), CInvA) + self._LInv

        # Compute the conditional mean and covariance
        cho_W = cho_factor(W)
        M = cho_solve(cho_W, tt.transpose(CInvA))
        ymu = tt.dot(M, flux - baseline_mean) + cho_solve(cho_W, self._LInvmu)
        ycov = cho_solve(cho_W, tt.eye(cho_W.shape[0]))
        cho_ycov = cho_factor(ycov)

        # Sample from it
        u = self.random.normal((self._nylm, nsamples))
        return tt.transpose(ymu[:, None] + tt.dot(cho_ycov, u))

    def mean(self, t, i=defaults["i"], p=defaults["p"]):
        if self._normalized:
            return tt.zeros_like(cast(t, vectorize=True))
        else:
            return self.flux.mean(t, i, p)

    def cov(self, t, i=defaults["i"], p=defaults["p"]):
        if self._normalized:
            cov = self.flux.cov(t, i, p)
            mean = self.flux.mean(t, i, p)
            return cov / (1 + mean) ** 2
        else:
            return self.flux.cov(t, i, p)

    def sample(self, t, i=defaults["i"], p=defaults["p"], nsamples=1):
        if self._marginalize_over_inclination:
            t = cast(t)
            u = self.random.normal((t.shape[0], nsamples))
            cho_cov = cho_factor(self.cov(t, i, p))
            return tt.transpose(
                self.mean(t, i, p)[:, None] + tt.dot(cho_cov, u)
            )
        else:
            ylm = self.sample_ylm(nsamples=nsamples)
            return tt.transpose(
                tt.dot(self.flux._design_matrix(t, i, p), tt.transpose(ylm))
            )

    def log_jac(self):
        """
        The log of the absolute value of the determinant of the Jacobian matrix.

        The spot latitude is Beta-distributed with shape parameters
        ``a`` and ``b``, equal to the log of the traditional ``alpha`` 
        and ``beta`` parameters of the Beta distribution, normalized and scaled
        to the range ``[0, 1]``. From Bayes' theorem, the joint posterior in 
        these two quantities is

            .. math::

                p\\big(a, b \\big| data\\big) \\sim 
                p\\big(data \\big| a, b\\big) \\times  p(a, b)

        However, this is a rather awkward parametrization, since it's hard to
        visualize how exactly ``a`` and ``b`` (or ``alpha`` and ``beta``) 
        determine quantities we actually care about, such as the mean ``mu`` and 
        standard deviation ``sigma`` of the distribution. This parameterization 
        is especially clumsy when it comes to specifying the prior ``p(a, b)``, 
        since any prior on these quantities will imply a very different prior 
        on ``mu`` and ``sigma``. In most cases, we probably want to place a prior 
        on ``mu`` and ``sigma`` directly. We can do this by noting that
        
            .. math::

                p(a, b) = p(\\mu, \\sigma) \\times J

        where

            .. math::

                J = \\bigg| \\frac{\\partial{\\mu}}{\\partial{a}} \\times 
                           \\frac{\\partial{\\sigma}}{\\partial{b}} -
                           \\frac{\\partial{\\mu}}{\\partial{b}} \\times 
                           \\frac{\\partial{\\sigma}}{\\partial{a}} \\bigg|

        is the absolute value of the determinant of the Jacobian matrix.

        Thus, to enforce a uniform prior on ``mu`` and ``sigma``, sample
        in ``a`` and ``b`` with a uniform prior in the range ``[0, 1]``
        and multiply the PDF by ``J``. Since we're in log space, you'll want 
        to add ``log J`` (the value returned by this function) to the
        log likelihood.

        """
        return self.latitude.log_jac()

    def log_likelihood(
        self,
        t,
        flux,
        data_cov,
        i=defaults["i"],
        p=defaults["p"],
        baseline_mean=defaults["baseline_mean"],
        baseline_var=defaults["baseline_var"],
    ):
        """
        Compute the log marginal likelihood of a light curve.

        Args:
            t (vector): The time array in arbitrary units.
            flux (vector): The array of observed flux values in arbitrary 
                units. In general, the flux should be either mean- or
                median-normalized with zero baseline. If the raw photometry
                is measured in ``counts``, users should compute the ``flux``
                from

                    .. code-block:: python
                    
                        flux = counts / np.median(counts) - 1

                If the baseline is something else (such as unity), users
                may alternatively set the ``baseline_mean`` parameter to
                reflect that.
                Note that if the ``normalized`` keyword passed to this class
                is ``False`` (not recommended for real data), then the flux
                should instead be normalized to the true baseline (i.e., the
                counts one would measure if the star had no spots). 
            data_cov (scalar, vector, or matrix): The data covariance 
                matrix. This may be a scalar equal to the (homoscedastic) 
                variance of the data, a vector equal to the variance of each 
                observation, or a matrix equal to the full covariance of the 
                dataset.
            i (scalar, optional): The inclination of the star in degrees.
                Default is %%defaults["i"]%%. If ``marginalize_over_inclination``
                is set, this argument is ignored.
            p (scalar, optional): The rotational period of the star in the same
                units as ``t``. Default is %%defaults["p"]%%.
            baseline_mean (scalar or vector, optional): The flux baseline to
                subtract when computing the GP likelihood. Default is 
                %%defaults["baseline_mean"]%%.
            baseline_var (scalar or matrix): The variance (square of the
                uncertainty) on the true value of the baseline. This is added
                to every element of the GP covariance matrix in order to
                marginalize over the baseline uncertainty. This may also be a
                matrix specifying the covariance due to additional correlated
                noise unrelated to star spot variability. Default is
                %%defaults["baseline_var"]%%.

        Returns:
            The log marginal likelihood of the `flux` vector conditioned on
            the current properties of the model. This is the likelihood 
            marginalized over all possible spherical harmonic vectors.

        """
        # Get the flux gp mean and covariance
        gp_mean = self.mean(t, i=i, p=p)
        gp_cov = self.cov(t, i=i, p=p)
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
