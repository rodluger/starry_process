from .. import logger
from ..math import cast, is_tensor
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as Beta
from scipy.special import beta as EulerBeta
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
import os
import theano.tensor as tt
from theano.ifelse import ifelse


# Get current path
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class BicubicSpline(object):

    STENCIL = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
            [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0],
            [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1],
            [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1],
            [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1],
            [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1],
        ]
    )

    def __init__(self, x, y, f):
        # Checks
        assert not is_tensor(x, y, f), "args cannot be theano nodes"
        assert x.ndim == y.ndim == 1
        assert f.ndim == 2

        # Add padding
        self.x = np.concatenate(
            (
                [-np.inf, -np.inf, 2 * x[0] - x[1]],
                x,
                [2 * x[-1] - x[-2], np.inf, np.inf],
            )
        )
        self.y = np.concatenate(
            (
                [-np.inf, -np.inf, 2 * y[0] - y[1]],
                y,
                [2 * y[-1] - y[-2], np.inf, np.inf],
            )
        )
        self.f = np.pad(f, 3, "edge")

        # Tensor versions
        self.x_tensor = cast(self.x)
        self.y_tensor = cast(self.y)
        self.f_tensor = cast(self.f)

    def __call__(self, x, y):

        # Choose the backend
        if is_tensor(x, y):
            math = tt
            func = self.f_tensor
            xarr = self.x_tensor
            yarr = self.y_tensor
        else:
            math = np
            func = self.f
            xarr = self.x
            yarr = self.y

        # Grid vertex just *below* (x, y)
        xi = math.argmin(x > xarr) - 1
        yi = math.argmin(y > yarr) - 1

        # Normalized distance to grid points
        dx = (x - xarr[xi]) / (xarr[xi + 1] - xarr[xi])
        dy = (y - yarr[yi]) / (yarr[yi + 1] - yarr[yi])

        # Function value and derivs at each vertex
        deriv = [
            func[xi, yi],
            func[xi + 1, yi],
            func[xi, yi + 1],
            func[xi + 1, yi + 1],
            0.5 * (func[xi + 1, yi] - func[xi - 1, yi]),
            0.5 * (func[xi + 2, yi] - func[xi, yi]),
            0.5 * (func[xi + 1, yi + 1] - func[xi - 1, yi + 1]),
            0.5 * (func[xi + 2, yi + 1] - func[xi, yi + 1]),
            0.5 * (func[xi, yi + 1] - func[xi, yi - 1]),
            0.5 * (func[xi + 1, yi + 1] - func[xi + 1, yi - 1]),
            0.5 * (func[xi, yi + 2] - func[xi, yi]),
            0.5 * (func[xi + 1, yi + 2] - func[xi + 1, yi]),
            0.25
            * (
                func[xi + 1, yi + 1]
                - func[xi - 1, yi + 1]
                - func[xi + 1, yi - 1]
                + func[xi - 1, yi - 1]
            ),
            0.25
            * (
                func[xi + 2, yi + 1]
                - func[xi, yi + 1]
                - func[xi + 2, yi - 1]
                + func[xi, yi - 1]
            ),
            0.25
            * (
                func[xi + 1, yi + 2]
                - func[xi - 1, yi + 2]
                - func[xi + 1, yi]
                + func[xi - 1, yi]
            ),
            0.25
            * (
                func[xi + 2, yi + 2]
                - func[xi, yi + 2]
                - func[xi + 2, yi]
                + func[xi, yi]
            ),
        ]

        # Dot with the stencil
        coeff = math.dot(self.STENCIL, deriv)
        dypow = 1
        ijkn = 0
        result = 0
        for j in range(4):
            result += dypow * (
                coeff[ijkn]
                + dx
                * (
                    coeff[ijkn + 1]
                    + dx * (coeff[ijkn + 2] + dx * coeff[ijkn + 3])
                )
            )
            ijkn += 4
            dypow *= dy

        return result


class Transform(object):
    def __init__(self, *args, **kwargs):
        pass

    def pdf(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def sample(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def transform(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")


class IdentityTransform(Transform):
    def transform(self, *args):
        return args

    def inverse_transform(self, *args):
        return args


class FixedTransform(IdentityTransform):
    def pdf(self, *args, **kwargs):
        raise NotImplementedError("This distribution does not have a PDF.")

    def sample(self, *args, **kwargs):
        raise NotImplementedError("This distribution cannot be sampled from.")


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

    _extra_arrays = {}

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
        return os.path.join(CACHE_PATH, self._name + "_" + self._hash + ".npz")

    @property
    def _arrays(self):
        return [
            "_mu_coeffs",
            "_mu_min",
            "_mu_max",
            "_sigma_coeffs",
            "_sigma_min",
            "_sigma_max",
            "_xp",
            "_y1p",
            "_y2p",
            "_lnalpha",
            "_lnbeta",
            "_logjac_arr",
        ] + list(sorted(self._extra_arrays.keys()))

    def _save(self):
        """
        Save the contents of this class to disk.
        
        """
        kwargs = {}
        for name in self._arrays:
            kwargs.update({name: getattr(self, name)})
        np.savez(self._cache_file, **kwargs)

    def _load(self):
        """
        Load the contents of this class from disk.
        
        """
        if os.path.exists(self._cache_file):
            # Load the arrays
            data = np.load(self._cache_file)
            for name in self._arrays:
                setattr(self, name, data[name])

            # Instantiate the interpolants
            self._sigma_min_func = interp1d(
                self._xp, self._y1p, kind="cubic", fill_value="extrapolate"
            )
            self._sigma_max_func = interp1d(
                self._xp, self._y2p, kind="cubic", fill_value="extrapolate"
            )
            self._logjac_spline = BicubicSpline(
                self._lnbeta, self._lnalpha, self._logjac_arr
            )

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

    def _ab_to_alphabeta(self, a, b):
        a1 = self._ln_alpha_min
        a2 = self._ln_alpha_max
        if is_tensor(a):
            alpha = tt.exp(cast(a1 + a * (a2 - a1)))
        else:
            alpha = np.exp(a1 + a * (a2 - a1))
        b1 = self._ln_beta_min
        b2 = self._ln_beta_max
        if is_tensor(b):
            beta = tt.exp(cast(b1 + b * (b2 - b1)))
        else:
            beta = np.exp(b1 + b * (b2 - b1))
        return alpha, beta

    def inverse_transform(self, a, b):
        """
        Return the mean and std. dev given `a` and `b`.
        
        """
        if is_tensor(a, b):
            raise NotImplementedError(
                "Inverse transform not implemented for tensor types."
            )
        a1 = self._ln_alpha_min
        a2 = self._ln_alpha_max
        b1 = self._ln_beta_min
        b2 = self._ln_beta_max
        alpha = np.exp(a * (a2 - a1) + a1)
        beta = np.exp(b * (b2 - b1) + b1)
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
        self._a = np.linspace(0, 1, self._mom_grid_res)
        self._b = np.linspace(0, 1, self._mom_grid_res)
        self._a, self._b = np.meshgrid(self._a, self._b)
        self._a = self._a.reshape(-1)
        self._b = self._b.reshape(-1)
        a1 = self._ln_alpha_min
        a2 = self._ln_alpha_max
        b1 = self._ln_beta_min
        b2 = self._ln_beta_max
        self._lnalpha = self._a * (a2 - a1) + a1
        self._lnbeta = self._b * (b2 - b1) + b1
        alpha = np.exp(self._lnalpha)
        beta = np.exp(self._lnbeta)
        beta_mu = alpha / (alpha + beta)
        beta_sigma = np.sqrt(
            alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        )

        # Compute the mu and sigma dev
        mu = np.empty_like(alpha)
        sigma = np.empty_like(alpha)
        for k in tqdm(range(len(alpha))):
            mu[k], sigma[k] = self.inverse_transform(self._a[k], self._b[k])

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
        self._xp = (
            (x + self._sigma_lim_tol)
            * (self._mu_max - self._mu_min)
            / (self._mu_max - self._mu_min + 2 * self._sigma_lim_tol)
        )
        self._y1p = y1 + self._sigma_lim_tol
        self._y2p = y2 - self._sigma_lim_tol

        # Get the interpolant
        self._sigma_min_func = interp1d(
            self._xp, self._y1p, kind="cubic", fill_value="extrapolate"
        )
        self._sigma_max_func = interp1d(
            self._xp, self._y2p, kind="cubic", fill_value="extrapolate"
        )

        # Fit a bivariate polynomial to the grid data
        x1 = (mu - self._mu_min) / (self._mu_max - self._mu_min)
        x2 = (sigma - self._sigma_min) / (self._sigma_max - self._sigma_min)
        A = self._get_BiV(x1, x2)
        self._mu_coeffs = np.linalg.solve(A.T @ A, A.T @ beta_mu)
        self._sigma_coeffs = np.linalg.solve(A.T @ A, A.T @ beta_sigma)

        # Now compute the derivatives
        dmu = np.array(
            np.gradient(mu.reshape(self._mom_grid_res, self._mom_grid_res))
        )
        dsigma = np.array(
            np.gradient(sigma.reshape(self._mom_grid_res, self._mom_grid_res))
        )
        dlnalpha = np.gradient(
            self._lnalpha.reshape(self._mom_grid_res, self._mom_grid_res)
        )[1]
        dlnbeta = np.gradient(
            self._lnbeta.reshape(self._mom_grid_res, self._mom_grid_res)
        )[0]
        alpha = np.exp(
            self._lnalpha.reshape(self._mom_grid_res, self._mom_grid_res)
        )
        beta = np.exp(
            self._lnbeta.reshape(self._mom_grid_res, self._mom_grid_res)
        )
        dmda = dmu[1] / dlnalpha / alpha
        dmdb = dmu[0] / dlnbeta / beta
        dsda = dsigma[1] / dlnalpha / alpha
        dsdb = dsigma[0] / dlnbeta / beta

        # Compute log abs det jac
        self._logjac_arr = np.log(np.abs(dmda * dsdb - dmdb * dsda))

        # Keep only the arrays
        self._lnbeta = self._lnbeta.reshape(
            self._mom_grid_res, self._mom_grid_res
        )[:, 0]
        self._lnalpha = self._lnalpha.reshape(
            self._mom_grid_res, self._mom_grid_res
        )[0]

        # Compute the interpolant
        self._logjac_spline = BicubicSpline(
            self._lnbeta, self._lnalpha, self._logjac_arr
        )

    def log_jac(self, a, b):
        """
        Returns the log of the abs value of the determinant of the Jacobian
        for this transformation.

        """
        # Jacobian for (a, b) <--> (alpha, beta) transform
        a1 = self._ln_alpha_min
        a2 = self._ln_alpha_max
        b1 = self._ln_beta_min
        b2 = self._ln_beta_max
        log_alpha = a * (a2 - a1) + a1
        log_beta = b * (b2 - b1) + b1
        log_jac = np.log(a2 - a1) + log_alpha + np.log(b2 - b1) + log_beta

        # Jacobian for (alpha, beta) <--> (mu, sigma) transform
        log_jac += self._logjac_spline(log_beta, log_alpha)

        return log_jac

    def transform(self, mu, sigma):
        """
        Return the `a` and `b` (scaled) parameters of the Beta distribution
        corresponding to a given `mu` and `sigma`.
        
        """
        # Theano-friendly implementation of this function
        if is_tensor(mu, sigma):
            return self._transform_tensor(mu, sigma)

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

        # Convert to standard params of the Beta distribution
        alpha = (beta_mu / beta_var) * ((1 - beta_mu) * beta_mu - beta_var)
        beta = beta_mu + (beta_mu / beta_var) * (1 - beta_mu) ** 2 - 1

        # Scale them to the [0, 1] range
        a1 = self._ln_alpha_min
        a2 = self._ln_alpha_max
        a = (np.log(alpha) - a1) / (a2 - a1)
        b1 = self._ln_beta_min
        b2 = self._ln_beta_max
        b = (np.log(beta) - b1) / (b2 - b1)
        return a, b

    def _transform_tensor(self, mu, sigma):

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
            ifelse(tt.gt(sigma, y1[tt.argmin(tt.abs_(x1 - mu))]), 0.0, np.nan),
            np.nan,
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

        # Scale them to the [0, 1] range
        a1 = self._ln_alpha_min
        a2 = self._ln_alpha_max
        a = (tt.log(alpha) - a1) / (a2 - a1)
        b1 = self._ln_beta_min
        b2 = self._ln_beta_max
        b = (tt.log(beta) - b1) / (b2 - b1)

        # Make NAN if bounds error
        a += nan_if_bounds_error
        b += nan_if_bounds_error

        return a, b

    def pdf(self, x, a, b):
        """
        Return the probability density function evaluated at `x`.
        
        """
        a1 = self._ln_alpha_min
        a2 = self._ln_alpha_max
        b1 = self._ln_beta_min
        b2 = self._ln_beta_max
        alpha = np.exp(a * (a2 - a1) + a1)
        beta = np.exp(b * (b2 - b1) + b1)

        # Easy!
        return self._jac(x) * Beta.pdf(self._f(x), alpha, beta)

    def sample(self, a, b, nsamples=1):
        """
        Draw samples from the distribution.
        
        """
        a1 = self._ln_alpha_min
        a2 = self._ln_alpha_max
        b1 = self._ln_beta_min
        b2 = self._ln_beta_max
        alpha = np.exp(a * (a2 - a1) + a1)
        beta = np.exp(b * (b2 - b1) + b1)

        # Sample
        x = Beta.rvs(alpha, beta, size=nsamples)
        return self._finv(x)
