from .wigner import R
from .integrals import WignerIntegral
from .ops import LatitudeIntegralOp, CheckBoundsOp
from .defaults import defaults
import theano.tensor as tt
from theano.ifelse import ifelse
from scipy.stats import beta as Beta
import numpy as np


def gauss2beta(
    mu,
    sigma,
    log_alpha_max=defaults["log_alpha_max"],
    log_beta_max=defaults["log_beta_max"],
    angle_unit=defaults["angle_unit"],
):
    """
    Return the shape parameters `a` and `b` of the latitude Beta distribution
    closest to the Gaussian with mean `mu` and standard deviation `sigma`.

    .. note:: 
        
        This is a utility function that accepts and returns numeric values
        (not tensors).

    """
    if angle_unit.startswith("deg"):
        angle_fac = np.pi / 180
    elif angle_unit.startswith("rad"):
        angle_fac = 1.0
    else:
        raise ValueError("Invalid `angle_unit`.")
    m = np.atleast_1d(mu) * angle_fac
    v = (np.atleast_1d(sigma) * angle_fac) ** 2
    c1 = np.cos(m)
    c2 = np.cos(2 * m)
    c3 = np.cos(3 * m)
    term = 1.0 / (16 * v * np.cos(0.5 * m) ** 4)
    alpha = (2 + 4 * v + (3 + 8 * v) * c1 + 2 * c2 + c3) * term
    beta = (c1 + 2 * v * (3 + c2) - c3) * term
    a = np.log(alpha) / log_alpha_max
    b = (np.log(beta) - np.log(0.5)) / (log_beta_max - np.log(0.5))
    return a, b


def beta2gauss(
    a,
    b,
    log_alpha_max=defaults["log_alpha_max"],
    log_beta_max=defaults["log_beta_max"],
    angle_unit=defaults["angle_unit"],
):
    """
    Return the mode `mu` and standard deviation `sigma` of Laplace's
    (Gaussian) approximation to the PDF of the latitude Beta distribution 
    with shape parameters `a` and `b`.

    .. note:: 
        
        This is a utility function that accepts and returns numeric values
        (not tensors).

    """
    if angle_unit.startswith("deg"):
        angle_fac = np.pi / 180
    elif angle_unit.startswith("rad"):
        angle_fac = 1.0
    else:
        raise ValueError("Invalid `angle_unit`.")
    alpha = np.atleast_1d(np.exp(a * log_alpha_max))
    beta = np.atleast_1d(
        np.exp(np.log(0.5) + b * (log_beta_max - np.log(0.5)))
    )
    term = (
        4 * alpha ** 2
        - 8 * alpha
        - 6 * beta
        + 4 * alpha * beta
        + beta ** 2
        + 5
    )
    mu = 2 * np.arctan(np.sqrt(2 * alpha + beta - 2 - np.sqrt(term)))
    term = (
        1
        - alpha
        + beta
        + (beta - 1) * np.cos(mu)
        + (alpha - 1) / np.cos(mu) ** 2
    )
    sigma = np.sin(mu) / np.sqrt(term)
    mu[(alpha <= 1) | (beta <= 0.5)] = np.nan
    sigma[(alpha <= 1) | (beta <= 0.5)] = np.nan
    return mu / angle_fac, sigma / angle_fac


class LatitudeIntegral(WignerIntegral):
    def _ingest(self, a, b, **kwargs):
        """
        Ingest the parameters of the distribution and 
        set up the transform and rotation operators.

        """
        # Ingest
        self._a = CheckBoundsOp(name="a", lower=0, upper=1, inclusive=False)(a)
        self._b = CheckBoundsOp(name="b", lower=0, upper=1, inclusive=False)(b)
        self._params = [self._a, self._b]

        # Transform to the shape parameters of the Beta distribution.
        # alpha is bounded on (1.0, exp(log_alpha_max))
        # beta is bounded on (0.5, exp(log_beta_max))
        self._log_alpha_max = kwargs.get(
            "log_alpha_max", defaults["log_alpha_max"]
        )
        self._log_beta_max = kwargs.get(
            "log_beta_max", defaults["log_beta_max"]
        )
        self._sigma_max = (
            kwargs.get("sigma_max", defaults["sigma_max"]) * self._angle_fac
        )
        self._alpha = tt.exp(self._a * self._log_alpha_max)
        self._beta = tt.exp(
            np.log(0.5) + self._b * (self._log_beta_max - np.log(0.5))
        )
        self._compute_mu_and_sigma()

        # Set up the rotation operator
        self._R = R(
            self._ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )

        # Compute the integrals
        self._integral_op = LatitudeIntegralOp(self._ydeg, **kwargs)
        self._q, _, _, self._Q, _, _ = self._integral_op(
            self._alpha, self._beta
        )

    @property
    def mu(self):
        return self._mu * self._angle_fac

    @property
    def sigma(self):
        return self._sigma * self._angle_fac

    def _compute_mu_and_sigma(self):
        term = (
            4 * self._alpha ** 2
            - 8 * self._alpha
            - 6 * self._beta
            + 4 * self._alpha * self._beta
            + self._beta ** 2
            + 5
        )
        mu = 2 * tt.arctan(
            tt.sqrt(2 * self._alpha + self._beta - 2 - tt.sqrt(term))
        )
        term = (
            1
            - self._alpha
            + self._beta
            + (self._beta - 1) * tt.cos(mu)
            + (self._alpha - 1) / tt.cos(mu) ** 2
        )
        var = tt.sin(mu) ** 2 / term
        self._mu = mu
        self._sigma = tt.sqrt(var)

    def _pdf(self, phi, a, b):
        """
        Return the probability density function evaluated at a 
        latitude `phi`.
        
        .. note:: 
        
            This function operates on and returns numeric values.
            It is used internally in the `perform` step of a `PDFOp`.

        """
        alpha = np.exp(a * self._log_alpha_max)
        beta = np.exp(np.log(0.5) + b * (self._log_beta_max - np.log(0.5)))
        phi = np.array(phi) * self._angle_fac
        return (
            0.5
            * np.abs(np.sin(phi) * self._angle_fac)
            * Beta.pdf(np.cos(phi), alpha, beta)
        )

    def _sample(self, a, b, nsamples=1):
        """
        Draw samples from the latitude distribution (in degrees).

        .. note:: 
        
            This function operates on and returns numeric values.
            It is used internally in the `perform` step of a `SampleOp`.
        
        """
        alpha = np.exp(a * self._log_alpha_max)
        beta = np.exp(np.log(0.5) + b * (self._log_beta_max - np.log(0.5)))
        x = Beta.rvs(alpha, beta, size=nsamples)
        return np.arccos(x) / self._angle_fac

    def _log_jac(self):
        """
        Return the log of the absolute value of the Jacobian for the
        transformation from `(a, b)` to `(mu, sigma)`.
        
        """
        log_jac = tt.log(
            tt.abs_(
                (
                    self._alpha
                    * self._beta
                    * (1 + tt.cos(self._mu)) ** 3
                    * tt.sin(2 * self._mu) ** 3
                )
                / (
                    self._sigma
                    * (
                        -3
                        + 2 * self._alpha
                        + self._beta
                        + (-1 + 2 * self._alpha + self._beta)
                        * tt.cos(self._mu)
                    )
                    * (
                        2 * (-1 + self._alpha + self._beta)
                        + 3 * (-1 + self._beta) * tt.cos(self._mu)
                        - 2
                        * (-1 + self._alpha - self._beta)
                        * tt.cos(2 * self._mu)
                        + (-1 + self._beta) * tt.cos(3 * self._mu)
                    )
                    ** 2
                )
            )
        )
        return ifelse(tt.gt(self._sigma, self._sigma_max), -np.inf, log_jac)
