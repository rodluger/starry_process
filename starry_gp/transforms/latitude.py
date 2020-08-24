from .transforms import Transform
from ..utils import logger
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as Beta
from scipy.special import beta as EulerBeta
from scipy.special import legendre
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
import os


__all__ = ["LatitudeTransform"]


# Get current path
PATH = os.path.dirname(os.path.abspath(__file__))


def get_moment(alpha, beta, n):
    """
    Compute the `nth` moment of the distribution of latitude given `alpha` and `beta`.
    
    """
    # We'll catch integration warnings
    warnings.filterwarnings("error", category=IntegrationWarning)

    # Normalization factor
    fac = 1.0 / EulerBeta(alpha, beta)

    def f(x):
        return (
            fac * np.arccos(x) ** n * x ** (alpha - 1) * (1 - x) ** (beta - 1)
        )

    def f_reparam(y, alpha, beta, reverse=False, min_ln=-100):
        if y < min_ln:
            beta_term = 1 - (beta - 1) * np.exp(y)
        else:
            beta_term = (1 - np.exp(y)) ** (beta - 1)
        if reverse:
            x = 1 - np.exp(y)
        else:
            x = np.exp(y)
        return fac * np.arccos(x) ** n * np.exp(y * alpha) * beta_term

    try:
        # Attempt to take the integral directly
        res, err = quad(f, 0, 1)

    except IntegrationWarning:
        # The Beta distribution can be tricky to integrate
        # when one or both of the parameters are really small,
        # since the dynamic range of the pdf is huge.
        # We can get better performance by splitting the integral
        # into two parts and substituting y = ln(x) in the left
        # half and y = ln(1 - x) in the right half.
        warnings.resetwarnings()
        res1, err1 = quad(
            f_reparam, -np.inf, np.log(0.5), args=(alpha, beta, False)
        )
        res2, err2 = quad(
            f_reparam, -np.inf, np.log(0.5), args=(beta, alpha, True)
        )
        res = res1 + res2
        err = err1 + err2

    # Reset the filter
    warnings.resetwarnings()

    return res


def get_latitude_mean_std(alpha, beta):
    """
    Return the mean and std. dev of the latitude distribution given `alpha` and `beta`.
    
    """
    mean = get_moment(alpha, beta, 1)
    std = np.sqrt(get_moment(alpha, beta, 2) - mean ** 2)
    return mean * 180 / np.pi, std * 180 / np.pi


def get_A(x1, x2, poly_order):
    """
    Return a bivariate Vandermonde design matrix.
    
    """
    A = np.ones_like(x1).reshape(-1, 1)
    for n in range(1, poly_order + 1):
        for k in range(n + 1):
            A = np.hstack((A, (x1 ** (n - k) * x2 ** k).reshape(-1, 1)))
    return A


def get_latitude_coeffs(**kwargs):

    logger.info("Computing latitude pdf transform...")

    # Fine-tuning params
    mom_grid_res = kwargs.get("mom_grid_res", 100)
    max_std = kwargs.get("max_std", 99)
    ln_alpha_min = kwargs.get("ln_alpha_min", -5.0)
    ln_alpha_max = kwargs.get("ln_alpha_max", 5.0)
    ln_beta_min = kwargs.get("ln_beta_min", -5.0)
    ln_beta_max = kwargs.get("ln_beta_max", 5.0)
    std_lim_tol = kwargs.get("std_lim_tol", 1.5)
    poly_order = kwargs.get("poly_order", 10)

    # Grid of Beta params
    lnalpha = np.linspace(ln_alpha_min, ln_alpha_max, mom_grid_res)
    lnbeta = np.linspace(ln_beta_min, ln_beta_max, mom_grid_res)
    lnalpha, lnbeta = np.meshgrid(lnalpha, lnbeta)
    lnalpha = lnalpha.reshape(-1)
    lnbeta = lnbeta.reshape(-1)
    alpha = np.exp(lnalpha)
    beta = np.exp(lnbeta)
    beta_mean = alpha / (alpha + beta)
    beta_std = np.sqrt(
        alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    )

    # Compute the mean and std dev of lat
    lat_mean = np.empty_like(alpha)
    lat_std = np.empty_like(alpha)
    for k in tqdm(range(len(alpha))):
        lat_mean[k], lat_std[k] = get_latitude_mean_std(alpha[k], beta[k])

    # Global max and min values for each
    lat_mean_lims = (np.min(lat_mean), np.max(lat_mean))
    lat_std_lims = (np.min(lat_std), np.max(lat_std))

    # The limits of lat std. dev. depend on the lat mean.
    # Get the minimum (y1) and maximum (y2) lat_std at each value of lat_mean
    x = np.linspace(lat_mean_lims[0], lat_mean_lims[1], mom_grid_res // 3)
    y1 = np.zeros_like(x) * np.nan
    y2 = np.zeros_like(x) * np.nan
    dx = x[1] - x[0]
    for k in range(len(x)):
        vals = lat_std[np.abs(lat_mean - x[k]) < 0.5 * dx]
        if len(vals):
            y2[k] = np.max(vals[vals < max_std])
            y1[k] = np.min(vals)

    # Add a little padding
    a, b = lat_mean_lims
    xp = (x + std_lim_tol) * (b - a) / (b - a + 2 * std_lim_tol)
    y1p = y1 + std_lim_tol
    y2p = y2 - std_lim_tol

    # Get the interpolant
    y1 = interp1d(xp, y1p, kind="cubic", fill_value="extrapolate")
    y2 = interp1d(xp, y2p, kind="cubic", fill_value="extrapolate")

    # Fit a bivariate polynomial to the grid data
    x1 = (lat_mean - lat_mean_lims[0]) / (lat_mean_lims[1] - lat_mean_lims[0])
    x2 = (lat_std - lat_std_lims[0]) / (lat_std_lims[1] - lat_std_lims[0])
    A = get_A(x1, x2, poly_order)
    lat_mean_coeffs = np.linalg.solve(A.T @ A, A.T @ beta_mean)
    lat_std_coeffs = np.linalg.solve(A.T @ A, A.T @ beta_std)

    return (
        lat_mean_coeffs,
        lat_mean_lims[0],
        lat_mean_lims[1],
        lat_std_coeffs,
        lat_std_lims[0],
        lat_std_lims[1],
        y1,
        y2,
    )


class LatitudeTransform(Transform):
    """
    Class hosting variable transforms for the spot size distribution.
    
    """

    def __init__(self, clobber=False, **kwargs):
        # Store the main ones
        self._poly_order = kwargs.get("poly_order", 10)
        self._mom_grid_res = kwargs.get("mom_grid_res", 100)

        # Get the kwargs hash
        self._hash = self._get_hash(**kwargs)

        # Attempt to load cached params from disk
        if clobber or not self._load():

            # Get the Beta distribution transform coeffs
            (
                self._mean_coeffs,
                self._mean_min,
                self._mean_max,
                self._std_coeffs,
                self._std_min,
                self._std_max,
                self._std_min_func,
                self._std_max_func,
            ) = get_latitude_coeffs(**kwargs)

            # Save!
            self._save()

    def _save(self):
        """
        Save the contents of this class to disk.
        
        """
        x = np.concatenate(
            (
                self._mean_coeffs,
                np.atleast_1d(self._mean_min),
                np.atleast_1d(self._mean_max),
                self._std_coeffs,
                np.atleast_1d(self._std_min),
                np.atleast_1d(self._std_max),
                self._std_min_func.x,
                self._std_min_func.y,
                self._std_max_func.x,
                self._std_max_func.y,
            )
        )
        np.savetxt(os.path.join(PATH, self._hash + ".dat"), x)

    def _load(self):
        """
        Load the contents of this class from disk.
        
        """
        if os.path.exists(os.path.join(PATH, self._hash + ".dat")):
            x = np.loadtxt(os.path.join(PATH, self._hash + ".dat"))
            ncoeffs = 0
            for i in range(self._poly_order + 1):
                for j in range(i + 1):
                    ncoeffs += 1
            self._mean_coeffs, x = np.split(x, [ncoeffs])
            self._mean_min, x = np.split(x, [1])
            self._mean_min = self._mean_min[0]
            self._mean_max, x = np.split(x, [1])
            self._mean_max = self._mean_max[0]
            self._std_coeffs, x = np.split(x, [ncoeffs])
            self._std_min, x = np.split(x, [1])
            self._std_min = self._std_min[0]
            self._std_max, x = np.split(x, [1])
            self._std_max = self._std_max[0]
            _x, x = np.split(x, [self._mom_grid_res // 3])
            _y, x = np.split(x, [self._mom_grid_res // 3])
            self._std_min_func = interp1d(
                _x, _y, kind="cubic", fill_value="extrapolate"
            )
            _x, x = np.split(x, [self._mom_grid_res // 3])
            _y, x = np.split(x, [self._mom_grid_res // 3])
            self._std_max_func = interp1d(
                _x, _y, kind="cubic", fill_value="extrapolate"
            )
            return True
        else:
            return False

    def _get_hash(self, **kwargs):
        """
        Return a hash string representation of the input kwargs.
        
        """
        mom_grid_res = kwargs.get("mom_grid_res", 100)
        max_std = kwargs.get("max_std", 99)
        ln_alpha_min = kwargs.get("ln_alpha_min", -5.0)
        ln_alpha_max = kwargs.get("ln_alpha_max", 5.0)
        ln_beta_min = kwargs.get("ln_beta_min", -5.0)
        ln_beta_max = kwargs.get("ln_beta_max", 5.0)
        std_lim_tol = kwargs.get("std_lim_tol", 1.50)
        poly_order = kwargs.get("poly_order", 10)
        params = [
            mom_grid_res,
            max_std,
            ln_alpha_min,
            ln_alpha_max,
            ln_beta_min,
            ln_beta_max,
            std_lim_tol,
            poly_order,
        ]
        return hex(
            int(
                "".join(["{:.0f}".format(abs(param) * 10) for param in params])
            )
        )

    def get_standard_params(self, mean, std):
        """
        Return the `alpha` and `beta` parameters of the Beta distribution
        corresponding to a given lat `mean` and `std`.
        
        """
        # Bounds checks
        mean = np.array(mean)
        std = np.array(std)
        assert np.all(
            (mean > self._mean_min) & (mean < self._mean_max)
        ), "mean is out of bounds"
        assert np.all(
            (std > self._std_min) & (std < self._std_max)
        ), "std is out of bounds"
        assert np.all(
            (std > self._std_min_func(mean)) & (std < self._std_max_func(mean))
        ), "std is out of bounds"

        # Linear fit
        x1 = (mean.reshape(-1) - self._mean_min) / (
            self._mean_max - self._mean_min
        )
        x2 = (std.reshape(-1) - self._std_min) / (
            self._std_max - self._std_min
        )
        A = get_A(x1, x2, self._poly_order)

        # Beta mean and variance
        beta_mean = (A @ self._mean_coeffs).reshape(mean.shape)
        beta_var = ((A @ self._std_coeffs).reshape(std.shape)) ** 2

        # Convert to standard params
        alpha = (beta_mean / beta_var) * (
            (1 - beta_mean) * beta_mean - beta_var
        )
        beta = beta_mean + (beta_mean / beta_var) * (1 - beta_mean) ** 2 - 1
        return alpha, beta

    def pdf(self, lat, mean, std):
        """
        Return the probability density function evaluated at `lat`.
        
        """
        # Transform to the standard params
        alpha, beta = self.get_standard_params(mean, std)

        # Get p(cos(lat))
        p_coslat = Beta.pdf(np.cos(lat * np.pi / 180), alpha, beta)

        # Compute the Jacobian
        jac = 0.5 * np.abs(np.sin(lat * np.pi / 180)) * np.pi / 180

        # We're done
        return jac * p_coslat

    def get_transform_error(self, res=100, plot=True):
        """
        Compute (and optionally plot) the empirical error in the transform
        between `alpha` and `beta` and the lat mean and standard deviation
        on a grid of resolution `res`.

        """
        # Compute the errors on a grid
        std = np.linspace(self._std_min, self._std_max, res)
        mean = np.linspace(self._mean_min, self._mean_max, res)
        mean_error = np.zeros((res, res))
        std_error = np.zeros((res, res))

        # Loop
        for i in tqdm(range(res)):
            for j in range(res):

                try:

                    # Get the error on the mean and std dev
                    val1, val2 = get_latitude_mean_std(
                        *self.get_standard_params(mean[j], std[i])
                    )
                    mean_error[i, j] = np.abs(val1 - mean[j]) / mean[j]
                    std_error[i, j] = np.abs(val2 - std[i]) / std[i]

                except AssertionError:

                    # We are out of bounds; not a problem
                    mean_error[i, j] = np.nan
                    std_error[i, j] = np.nan

        if plot:

            # Plot the results
            fig, ax = plt.subplots(
                1, 2, figsize=(12, 5), sharex=True, sharey=True
            )
            extent = (
                self._mean_min,
                self._mean_max,
                self._std_min,
                self._std_max,
            )
            im = ax[0].imshow(
                mean_error, origin="lower", extent=extent, aspect="auto"
            )
            plt.colorbar(im, ax=ax[0])
            im = ax[1].imshow(
                std_error, origin="lower", extent=extent, aspect="auto"
            )
            plt.colorbar(im, ax=ax[1])

            # Plot the empirical data boundary
            x = np.linspace(self._mean_min, self._mean_max, 100)
            for axis in ax:
                axis.plot(x, self._std_min_func(x), color="k")
                axis.plot(x, self._std_max_func(x), color="k")
                axis.fill_between(
                    x, self._std_max_func(x), self._std_max, color="w"
                )
                axis.fill_between(
                    x, self._std_min, self._std_min_func(x), color="w"
                )

            # Appearance
            ax[0].set_title("lat mean error")
            ax[1].set_title("lat std error")
            ax[0].set_xlim(self._mean_min, self._mean_max)
            ax[0].set_ylim(self._std_min, self._std_max)
            ax[0].set_xlabel("lat mean")
            ax[1].set_xlabel("lat mean")
            ax[0].set_ylabel("lat std")

            return mean, mean_error, std, std_error, fig, ax

        else:

            return mean, mean_error, std, std_error
