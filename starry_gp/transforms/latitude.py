from .transforms import Transform
from ..utils import logger
import numpy as np
from scipy.special import beta as EulerBeta
from scipy.special import betaln as LogEulerBeta


__all__ = ["LatitudeTransform"]


class LatitudeTransform(Transform):
    """
    Class hosting variable transforms for the spot latitude distribution.
    
    """

    def __init__(self, ydeg, **kwargs):
        self._ydeg = ydeg

    def get_standard_params(self, mean, std):
        """
        Return the `alpha` and `beta` parameters of the Beta distribution
        given the mean and standard deviation.
        
        """
        # Bounds checks
        mean = np.array(mean)
        std = np.array(std)
        assert np.all((mean > 0) & (mean < 1)), "mean is out of bounds"
        assert np.all((std > 0)), "std is out of bounds"

        # Transform
        var = std ** 2
        alpha = (mean / var) * ((1 - mean) * mean - var)
        beta = mean + (mean / var) * (1 - mean) ** 2 - 1
        return alpha, beta

    def pdf(self, phi, mean, std):
        """
        Return the probability density function evaluated at latitude `phi`.
        
        """
        # Transform to the standard params
        alpha, beta = self.get_standard_params(mean, std)

        x = np.cos(phi * np.pi / 180)
        y = np.abs(np.sin(phi * np.pi / 180))

        if np.any(alpha > 300) or np.any(beta > 300):
            logpdf = (
                (alpha - 1) * np.log(x)
                + (beta - 1) * np.log(1 - x)
                + np.log(0.5 * y)
                - LogEulerBeta(alpha, beta)
            )
            return np.exp(logpdf)
        else:
            return (
                1.0
                / EulerBeta(alpha, beta)
                * 0.5
                * y
                * x ** (alpha - 1)
                * (1 - x) ** (beta - 1)
            )
