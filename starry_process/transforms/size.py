from .. import logger
from .transforms import BetaTransform
import numpy as np
from scipy.special import legendre
from scipy.optimize import minimize


__all__ = ["SizeTransform"]


def _rhop_to_hwhm(rhop):
    """
    Return the theoretical half-witdth at half minimum as a function of rho'.
    
    """
    return (
        np.arccos((2 + 3 * rhop * (2 + rhop)) / (2 * (1 + rhop) ** 3))
        * 180
        / np.pi
    )


def _hwhm_to_rhop(hwhm):
    """
    Return rho' as a function of the theoretical half-width at half minimum.
    
    """
    theta = hwhm * np.pi / 180
    return (1 + np.cos(2 * theta / 3) + np.sqrt(3) * np.sin(2 * theta / 3)) / (
        2 * np.cos(theta)
    ) - 1


def _rhop_max(hwhm_max):
    """
    Returns the value of rho' corresponding to `hwhm_max`.
    
    """
    f = lambda rhop: (_rhop_to_hwhm(rhop) - hwhm_max) ** 2
    res = minimize(f, 2.0)
    return res.x[0]


def _corr(rhop, c):
    """Intensity correction function."""
    rho = (rhop - c[0]) / c[1]
    return 1 + c[2] * (1 - rho) ** c[3]


def _I(ydeg, rhop, theta, c=None):
    """
    Return the intensity at polar angle `theta` (in deg) away from
    the center of a spot of radius rho' expanded to degree `ydeg`.
    
    """
    # Compute the Legendre expansion
    cost = np.cos(theta * np.pi / 180)
    term = np.sum(
        [(1 + rhop) ** -l * legendre(l)(cost) for l in range(ydeg + 1)], axis=0
    )
    I = 0.5 * rhop * (1 - (2 + rhop) / (1 + rhop) * term)

    # Apply the intensity correction
    if c is not None:
        I *= _corr(rhop, c)

    return I


def _rhop_to_hwhm_empirical(ydeg, rhop):
    """
    Return the empirical half-width at half minimum as a function of rho'.
    
    """
    # Setup
    rhop = np.atleast_1d(rhop)
    hwhm_empirical = np.zeros_like(rhop)

    # Find the HWHM numerically for each radius
    for k in range(len(rhop)):

        halfmax = 0.5 * _I(ydeg, rhop[k], 0)

        def loss(theta):
            return (_I(ydeg, rhop[k], theta) - halfmax) ** 2

        res = minimize(loss, _rhop_to_hwhm(max(0.1, rhop[k])))
        hwhm_empirical[k] = res.x[0]

    return hwhm_empirical


def _get_c(ydeg, hwhm_max, hwhm_min, c_npts):
    """
    Return the coefficients for the radius transformation.

    """
    logger.info("Computing radius transform coefficients...")
    c = np.zeros(4)

    # Minimum rho': we need to optimize numerically
    loss = lambda p: (_rhop_to_hwhm_empirical(ydeg, p[0]) - hwhm_min) ** 2
    res = minimize(loss, _hwhm_to_rhop(hwhm_min))
    rhopmin = res.x[0]
    c[0] = rhopmin

    # Maximum rho' (easy)
    rhopmax = _rhop_max(hwhm_max)
    c[1] = rhopmax - rhopmin

    # Now compute the coefficients of the intensity
    # correction, c[2] and c[3].

    # Array over which to compute the loss
    rhop = np.linspace(rhopmin + 1e-6, rhopmax - 1e-6, c_npts)

    # Get the actual (absolute value of the) intensity at the peak
    l = np.arange(ydeg + 1).reshape(1, -1)
    term = np.sum((1 + rhop.reshape(-1, 1)) ** -l, axis=-1)
    I = -0.5 * rhop * (1 - (2 + rhop) / (1 + rhop) * term)

    # This is the factor by which we need to normalize the function
    norm = 1.0 / I

    # Find the coefficients of the fit (least squares)
    diff = lambda p: np.sum(
        (norm - _corr(rhop, [c[0], c[1], p[0], p[1]])) ** 2
    )
    res = minimize(diff, [0.1, 50.0])
    c[2:] = res.x

    # Log the deets
    logger.info(
        "Delta theta range: {:.2f} - {:.2f} degrees".format(hwhm_min, hwhm_max)
    )
    logger.info(
        "c coeffs: {:s}".format(" ".join(["{:.8f}".format(ck) for ck in c]))
    )
    logger.info(
        "Maximum intensity |error|: {:.2e}".format(
            np.max(np.abs(norm - _corr(rhop, c)))
        )
    )
    logger.info(
        "Average intensity |error|: {:.2e}".format(
            np.mean(np.abs(norm - _corr(rhop, c)))
        )
    )

    return c


class SizeTransform(BetaTransform):

    _name = "size"
    _defaults = {
        "mom_grid_res": 100,
        "max_sigma": 30.0,
        "ln_alpha_min": -5.0,
        "ln_alpha_max": 5.0,
        "ln_beta_min": -5.0,
        "ln_beta_max": 5.0,
        "sigma_lim_tol": 0.75,
        "poly_order": 10,
        "ydeg": 15,
        "hwhm_max": 75,
        "hwhm_min": 15,
        "c_npts": 100,
    }
    _extra_params = {"c": 4}

    def _f(self, x):
        theta = x * np.pi / 180
        rhop = (
            1 + np.cos(2 * theta / 3) + np.sqrt(3) * np.sin(2 * theta / 3)
        ) / (2 * np.cos(theta)) - 1
        return (rhop - self._c[0]) / self._c[1]

    def _jac(self, x):
        theta = x * np.pi / 180
        return np.abs(
            (
                2 * np.sqrt(3) * np.cos(theta / 3)
                - np.sqrt(3) * np.cos(theta)
                + 4 * np.sin(theta / 3)
                + np.sin(theta)
            )
            / (3 * self._c[1] * (1 - 2 * np.cos(2 * theta / 3)) ** 2)
            * np.pi
            / 180
        )

    def _finv(self, f_of_x):
        rhop = self._c[0] + self._c[1] * f_of_x
        return (
            np.arccos((2 + 3 * rhop * (2 + rhop)) / (2 * (1 + rhop) ** 3))
            * 180
            / np.pi
        )

    def _preprocess(self):
        self._c = _get_c(
            self._ydeg, self._hwhm_max, self._hwhm_min, self._c_npts
        )

    @property
    def c(self):
        return self._c

    def get_s(self, rho=None, rhoprime=None, hwhm=None, apply_correction=True):
        """
        Return the spot spherical harmonic expansion vector `s`.

        """
        assert (
            np.count_nonzero(
                [rho is not None, rhoprime is not None, hwhm is not None]
            )
            == 1
        ), "please provide exactly one of `rho`, `rhoprime`, `hwhm`."
        if rhoprime is None:
            if rho is None:
                rhoprime = self.hwhm_to_rhoprime(hwhm)
            else:
                rhoprime = self.rho_to_rhoprime(rho)

        rhoprime = np.atleast_1d(rhoprime)
        assert len(rhoprime.shape) == 1
        K = rhoprime.shape[0]
        sm0 = np.zeros((K, self._ydeg + 1))
        sm0[:, 0] = 0.5 * rhoprime
        for l in range(self._ydeg + 1):
            sm0[:, l] -= (
                rhoprime
                * (2 + rhoprime)
                / (2 * np.sqrt(2 * l + 1) * (1 + rhoprime) ** (l + 1))
            )
        if apply_correction:
            sm0 *= _corr(rhoprime, self._c).reshape(-1, 1)
        l = np.arange(self._ydeg + 1)
        s = np.zeros((K, (self._ydeg + 1) * (self._ydeg + 1)))
        s[:, l * (l + 1)] = sm0
        return s

    def rho_to_rhoprime(self, rho):
        """
        Return the transformed radius parameter, rho'.
        
        """
        return self._c[0] + self._c[1] * rho

    def rhoprime_to_hwhm(self, rhop):
        """
        Return the theoretical half-witdth at half minimum as a function of rho'.

        """
        return _rhop_to_hwhm(rhop)

    def hwhm_to_rhoprime(self, hwhm):
        """
        Return rho' as a function of the theoretical half-width at half minimum.

        """
        return _hwhm_to_rhop(hwhm)

    def get_intensity(
        self, theta, rho=None, rhoprime=None, hwhm=None, apply_correction=True
    ):
        """
        Return the intensity of a spot of radius `rho` at polar angle `theta`.

        """
        assert (
            np.count_nonzero(
                [rho is not None, rhoprime is not None, hwhm is not None]
            )
            == 1
        ), "please provide exactly one of `rho`, `rhoprime`, `hwhm`."
        if rhoprime is None:
            if rho is None:
                rhoprime = self.hwhm_to_rhoprime(hwhm)
            else:
                rhoprime = self.rho_to_rhoprime(rho)
        if apply_correction:
            return _I(self._ydeg, rhoprime, theta, c=self._c)
        else:
            return _I(self._ydeg, rhoprime, theta, c=None)
