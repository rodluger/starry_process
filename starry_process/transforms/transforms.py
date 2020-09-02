from .. import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as Beta
from scipy.special import beta as EulerBeta
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
import os
import theano
import theano.tensor as tt
from theano.ifelse import ifelse
from inspect import getmro


# Get current path
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def is_theano(*objs):
    """Return ``True`` if any of ``objs`` is a ``Theano`` object."""
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


class Transform(object):
    def __init__(self, *args, **kwargs):
        pass

    def pdf(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def sample(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def transform_params(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")


class IdentityTransform(Transform):
    def transform_params(self, *args):
        return args


class BetaTransform(Transform):

    _name = ""
    _defaults = {
        "mom_grid_res": 100,
        "max_sigma": 0,
        "ln_alpha_min": -5.0,
        "ln_alpha_max": 5.0,
        "ln_beta_min": -5.0,
        "ln_beta_max": 5.0,
        "sigma_lim_tol": 0.0,
        "poly_order": 10,
    }

    _extra_params = {}

    def __init__(self, clobber=False, **kwargs):
        # Load the kwargs
        self._params = []
        for key, value in self._defaults.items():
            setattr(self, "_{0}".format(key), kwargs.pop(key, value))
            self._params.append(getattr(self, "_{0}".format(key)))
        self._compute_hash()

        # Either load or compute the transform coeffs
        if clobber or not self._load():
            self._preprocess()
            self._compute_coeffs()
            self._save()

    @property
    def _cache_file(self):
        return os.path.join(CACHE_PATH, self._name + "_" + self._hash + ".dat")

    def _save(self):
        """
        Save the contents of this class to disk.
        
        """
        x = np.concatenate(
            (
                self._mu_coeffs,
                np.atleast_1d(self._mu_min),
                np.atleast_1d(self._mu_max),
                self._sigma_coeffs,
                np.atleast_1d(self._sigma_min),
                np.atleast_1d(self._sigma_max),
                self._sigma_min_func.x,
                self._sigma_min_func.y,
                self._sigma_max_func.x,
                self._sigma_max_func.y,
            )
        )
        for name, length in self._extra_params.items():
            x = np.concatenate((x, getattr(self, "_{0}".format(name))))
        np.savetxt(self._cache_file, x)

    def _load(self):
        """
        Load the contents of this class from disk.
        
        """
        if os.path.exists(self._cache_file):
            x = np.loadtxt(self._cache_file)
            ncoeffs = 0
            for i in range(self._poly_order + 1):
                for j in range(i + 1):
                    ncoeffs += 1
            self._mu_coeffs, x = np.split(x, [ncoeffs])
            self._mu_min, x = np.split(x, [1])
            self._mu_min = self._mu_min[0]
            self._mu_max, x = np.split(x, [1])
            self._mu_max = self._mu_max[0]
            self._sigma_coeffs, x = np.split(x, [ncoeffs])
            self._sigma_min, x = np.split(x, [1])
            self._sigma_min = self._sigma_min[0]
            self._sigma_max, x = np.split(x, [1])
            self._sigma_max = self._sigma_max[0]
            _x, x = np.split(x, [self._mom_grid_res // 3])
            _y, x = np.split(x, [self._mom_grid_res // 3])
            self._sigma_min_func = interp1d(
                _x, _y, kind="cubic", fill_value="extrapolate"
            )
            _x, x = np.split(x, [self._mom_grid_res // 3])
            _y, x = np.split(x, [self._mom_grid_res // 3])
            self._sigma_max_func = interp1d(
                _x, _y, kind="cubic", fill_value="extrapolate"
            )
            for name, length in self._extra_params.items():
                value, x = np.split(x, [length])
                setattr(self, "_{0}".format(name), value)
            return True
        else:
            return False

    def _compute_hash(self):
        """
        Return a hash string representation of the input kwargs.
        
        """
        self._hash = hex(
            int(
                "".join(
                    [
                        "{:.0f}".format(abs(param) * 100)
                        for param in self._params
                    ]
                )
            )
        )

    def _preprocess(self):
        # Subclass me to run code before computing the transform coeffs.
        pass

    def _f(self, x):
        raise NotImplementedError("Must be subclassed.")

    def _jac(self, x):
        raise NotImplementedError("Must be subclassed.")

    def _finv(self, f_of_x):
        raise NotImplementedError("Must be subclassed.")

    def _get_moment(self, alpha, beta, n):
        """
        Compute the `nth` moment of `_finv(x)` under a Beta distribution 
        with shape parameters `alpha` and `beta`.
        
        """
        # We'll catch integration warnings
        warnings.filterwarnings("error", category=IntegrationWarning)

        # Normalization factor
        fac = 1.0 / EulerBeta(alpha, beta)

        def integrand(f_of_x):
            return (
                fac
                * self._finv(f_of_x) ** n
                * f_of_x ** (alpha - 1)
                * (1 - f_of_x) ** (beta - 1)
            )

        def integrand_reparam(y, alpha, beta, reverse=False, min_ln=-100):
            if y < min_ln:
                beta_term = 1 - (beta - 1) * np.exp(y)
            else:
                beta_term = (1 - np.exp(y)) ** (beta - 1)
            if reverse:
                f_of_x = 1 - np.exp(y)
            else:
                f_of_x = np.exp(y)
            return (
                fac * self._finv(f_of_x) ** n * np.exp(y * alpha) * beta_term
            )

        try:
            # Attempt to take the integral directly
            res, err = quad(integrand, 0, 1)

        except IntegrationWarning:
            # The Beta distribution can be tricky to integrate
            # when one or both of the parameters are really small,
            # since the dynamic range of the pdf is huge.
            # We can get better performance by splitting the integral
            # into two parts and substituting y = ln(f_of_x) in the left
            # half and y = ln(1 - f_of_x) in the right half.
            warnings.resetwarnings()
            res1, err1 = quad(
                integrand_reparam,
                -np.inf,
                np.log(0.5),
                args=(alpha, beta, False),
            )
            res2, err2 = quad(
                integrand_reparam,
                -np.inf,
                np.log(0.5),
                args=(beta, alpha, True),
            )
            res = res1 + res2
            err = err1 + err2

        # Reset the filter
        warnings.resetwarnings()

        return res

    def _get_mu_sigma(self, alpha, beta):
        """
        Return the mu and sigma. dev given `alpha` and `beta`.
        
        """
        mu = self._get_moment(alpha, beta, 1)
        sigma = np.sqrt(self._get_moment(alpha, beta, 2) - mu ** 2)
        return mu, sigma

    def _get_BiV(self, x1, x2):
        """
        Return a bivariate Vandermonde design matrix.
        
        """
        A = np.ones_like(x1).reshape(-1, 1)
        for n in range(1, self._poly_order + 1):
            for k in range(n + 1):
                A = np.hstack((A, (x1 ** (n - k) * x2 ** k).reshape(-1, 1)))
        return A

    def _compute_coeffs(self):

        logger.info("Computing {0} pdf transform...".format(self._name))

        # Grid of Beta params
        lnalpha = np.linspace(
            self._ln_alpha_min, self._ln_alpha_max, self._mom_grid_res
        )
        lnbeta = np.linspace(
            self._ln_beta_min, self._ln_beta_max, self._mom_grid_res
        )
        lnalpha, lnbeta = np.meshgrid(lnalpha, lnbeta)
        lnalpha = lnalpha.reshape(-1)
        lnbeta = lnbeta.reshape(-1)
        alpha = np.exp(lnalpha)
        beta = np.exp(lnbeta)
        beta_mu = alpha / (alpha + beta)
        beta_sigma = np.sqrt(
            alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        )

        # Compute the mu and sigma dev
        mu = np.empty_like(alpha)
        sigma = np.empty_like(alpha)
        for k in tqdm(range(len(alpha))):
            mu[k], sigma[k] = self._get_mu_sigma(alpha[k], beta[k])

        # Global max and min values for each
        self._mu_min, self._mu_max = (np.min(mu), np.max(mu))
        self._sigma_min, self._sigma_max = (np.min(sigma), np.max(sigma))

        # The limits of sigma. dev. depend on the mu.
        # Get the minimum (y1) and maximum (y2) sigma at each value of mu
        x = np.linspace(self._mu_min, self._mu_max, self._mom_grid_res // 3)
        y1 = np.zeros_like(x) * np.nan
        y2 = np.zeros_like(x) * np.nan
        dx = x[1] - x[0]
        for k in range(len(x)):
            vals = sigma[np.abs(mu - x[k]) < 0.5 * dx]
            if len(vals):
                if self._max_sigma > 0:
                    y2[k] = np.max(vals[vals < self._max_sigma])
                else:
                    y2[k] = np.max(vals)
                y1[k] = np.min(vals)

        # Add a little padding
        xp = (
            (x + self._sigma_lim_tol)
            * (self._mu_max - self._mu_min)
            / (self._mu_max - self._mu_min + 2 * self._sigma_lim_tol)
        )
        y1p = y1 + self._sigma_lim_tol
        y2p = y2 - self._sigma_lim_tol

        # Get the interpolant
        self._sigma_min_func = interp1d(
            xp, y1p, kind="cubic", fill_value="extrapolate"
        )
        self._sigma_max_func = interp1d(
            xp, y2p, kind="cubic", fill_value="extrapolate"
        )

        # Fit a bivariate polynomial to the grid data
        x1 = (mu - self._mu_min) / (self._mu_max - self._mu_min)
        x2 = (sigma - self._sigma_min) / (self._sigma_max - self._sigma_min)
        A = self._get_BiV(x1, x2)
        self._mu_coeffs = np.linalg.solve(A.T @ A, A.T @ beta_mu)
        self._sigma_coeffs = np.linalg.solve(A.T @ A, A.T @ beta_sigma)

    def transform_params(self, mu, sigma):
        """
        Return the `alpha` and `beta` parameters of the Beta distribution
        corresponding to a given `mu` and `sigma`.
        
        """
        # Theano-friendly implementation of this function
        if is_theano(mu, sigma):
            return self._transform_params_theano(mu, sigma)

        # Bounds checks
        mu = np.array(mu)
        sigma = np.array(sigma)
        assert np.all(
            (mu > self._mu_min) & (mu < self._mu_max)
        ), "mu is out of bounds"
        assert np.all(
            (sigma > self._sigma_min) & (sigma < self._sigma_max)
        ), "sigma is out of bounds"
        assert np.all(
            (sigma > self._sigma_min_func(mu))
            & (sigma < self._sigma_max_func(mu))
        ), "sigma is out of bounds"

        # Linear fit
        x1 = (mu.reshape(-1) - self._mu_min) / (self._mu_max - self._mu_min)
        x2 = (sigma.reshape(-1) - self._sigma_min) / (
            self._sigma_max - self._sigma_min
        )
        A = self._get_BiV(x1, x2)

        # Beta mu and variance
        beta_mu = (A @ self._mu_coeffs).reshape(mu.shape)
        beta_var = ((A @ self._sigma_coeffs).reshape(sigma.shape)) ** 2

        # Convert to standard params
        alpha = (beta_mu / beta_var) * ((1 - beta_mu) * beta_mu - beta_var)
        beta = beta_mu + (beta_mu / beta_var) * (1 - beta_mu) ** 2 - 1
        return alpha, beta

    def _transform_params_theano(self, mu, sigma):

        # Bounds checks
        nan_if_bounds_error = ifelse(
            tt.lt(mu, self._mu_max),
            ifelse(tt.gt(mu, self._mu_min), 0.0, np.nan),
            np.nan,
        ) + ifelse(
            tt.lt(sigma, self._sigma_max),
            ifelse(tt.gt(sigma, self._sigma_min), 0.0, np.nan),
            np.nan,
        )

        # Nearest-neighbor check for `sigma_min(mu)` and `sigma_max(mu)``
        x1 = tt.as_tensor_variable(self._sigma_min_func.x)
        y1 = tt.as_tensor_variable(self._sigma_min_func.y)
        x2 = tt.as_tensor_variable(self._sigma_max_func.x)
        y2 = tt.as_tensor_variable(self._sigma_max_func.y)
        nan_if_bounds_error += ifelse(
            tt.lt(sigma, y2[tt.argmin(tt.abs_(x2 - mu))]),
            ifelse(
                tt.gt(sigma, y1[tt.argmin(tt.abs_(x1 - mu))]),
                0.0,
                np.nan
            ),
            np.nan
        )

        # Linear fit
        x1 = (mu - self._mu_min) / (self._mu_max - self._mu_min)
        x2 = (sigma - self._sigma_min) / (self._sigma_max - self._sigma_min)
        A = tt.ones(1 + self._poly_order * (self._poly_order + 3) // 2)
        i = 1
        for n in range(1, self._poly_order + 1):
            for k in range(n + 1):
                A = tt.set_subtensor(A[i], x1 ** (n - k) * x2 ** k)
                i += 1

        # Beta mu and variance
        beta_mu = tt.dot(A, self._mu_coeffs)
        beta_var = tt.dot(A, self._sigma_coeffs) ** 2

        # Convert to standard params
        alpha = (beta_mu / beta_var) * ((1 - beta_mu) * beta_mu - beta_var)
        beta = beta_mu + (beta_mu / beta_var) * (1 - beta_mu) ** 2 - 1

        # Make NAN if bounds error
        alpha += nan_if_bounds_error
        beta += nan_if_bounds_error

        return alpha, beta

    def pdf(self, x, mu=None, sigma=None, alpha=None, beta=None):
        """
        Return the probability density function evaluated at `x`.
        
        """
        assert ((mu is not None) and (sigma is not None)) or (
            (alpha is not None) and (beta is not None)
        ), "must provide either `mu` and `sigma` or `alpha` and `beta`"

        assert not (
            (mu is not None) and (alpha is not None)
        ), "cannot provide both `mu, sigma` and `alpha, beta`."

        # Transform to the standard params
        if alpha is None:
            alpha, beta = self.transform_params(mu, sigma)

        # Easy!
        return self._jac(x) * Beta.pdf(self._f(x), alpha, beta)

    def sample(self, mu=None, sigma=None, alpha=None, beta=None, nsamples=1):
        """
        Draw samples from the distribution.
        
        """
        assert ((mu is not None) and (sigma is not None)) or (
            (alpha is not None) and (beta is not None)
        ), "must provide either `mu` and `sigma` or `alpha` and `beta`"

        assert not (
            (mu is not None) and (alpha is not None)
        ), "cannot provide both `mu, sigma` and `alpha, beta`."

        # Transform to the standard params
        if alpha is None:
            alpha, beta = self.transform_params(mu, sigma)

        # Sample
        x = Beta.rvs(alpha, beta, size=nsamples)
        return self._finv(x)

    def get_transform_error(self, res=100, plot=True):
        """
        Compute (and optionally plot) the empirical error in the transform
        between `alpha` and `beta` and the mean and standard deviation
        on a grid of resolution `res`.

        """
        # Compute the errors on a grid
        sigma = np.linspace(self._sigma_min, self._sigma_max, res)
        mu = np.linspace(self._mu_min, self._mu_max, res)
        mu_error = np.zeros((res, res))
        sigma_error = np.zeros((res, res))

        # Loop
        for i in tqdm(range(res)):
            for j in range(res):

                try:

                    # Get the error on the mu and sigma dev
                    val1, val2 = self._get_mu_sigma(
                        *self.transform_params(mu[j], sigma[i])
                    )
                    mu_error[i, j] = np.abs(val1 - mu[j]) / mu[j]
                    sigma_error[i, j] = np.abs(val2 - sigma[i]) / sigma[i]

                except AssertionError:

                    # We are out of bounds; not a problem
                    mu_error[i, j] = np.nan
                    sigma_error[i, j] = np.nan

        if plot:

            # Plot the results
            fig, ax = plt.subplots(
                1, 2, figsize=(12, 5), sharex=True, sharey=True
            )
            extent = (
                self._mu_min,
                self._mu_max,
                self._sigma_min,
                self._sigma_max,
            )
            im = ax[0].imshow(
                mu_error, origin="lower", extent=extent, aspect="auto"
            )
            plt.colorbar(im, ax=ax[0])
            im = ax[1].imshow(
                sigma_error, origin="lower", extent=extent, aspect="auto"
            )
            plt.colorbar(im, ax=ax[1])

            # Plot the empirical data boundary
            x = np.linspace(self._mu_min, self._mu_max, 100)
            for axis in ax:
                axis.plot(x, self._sigma_min_func(x), color="k")
                axis.plot(x, self._sigma_max_func(x), color="k")
                axis.fill_between(
                    x, self._sigma_max_func(x), self._sigma_max, color="w"
                )
                axis.fill_between(
                    x, self._sigma_min, self._sigma_min_func(x), color="w"
                )

            # Appearance
            ax[0].set_title("mu error")
            ax[1].set_title("sigma error")
            ax[0].set_xlim(self._mu_min, self._mu_max)
            ax[0].set_ylim(self._sigma_min, self._sigma_max)
            ax[0].set_xlabel("mu")
            ax[1].set_xlabel("mu")
            ax[0].set_ylabel("sigma")

            return mu, mu_error, sigma, sigma_error, fig, ax

        else:

            return mu, mu_error, sigma, sigma_error
