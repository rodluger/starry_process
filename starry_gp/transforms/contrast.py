from .transforms import Transform
from ..utils import logger
import numpy as np


__all__ = ["ContrastTransform"]


class ContrastTransform(Transform):
    """
    Class hosting variable transforms for the spot contrast distribution.
    
    """

    def __init__(self, ydeg, **kwargs):
        self._ydeg = ydeg

    def get_standard_params(self, mean, std):
        """
        Return the mean and standard deviation of the log-normal distribution
        in *brightness*.
        
        """
        # Bounds checks
        mean = np.array(mean)
        std = np.array(std)
        assert np.all((std > 0)), "std is out of bounds"

        v = std ** 2
        b = (1 - mean) ** 2
        mu = np.log(b / np.sqrt(b + v))
        sigma = np.log(1 + v / b)
        return mu, sigma

    def pdf(self, xi, mean, std):
        """
        Return the probability density function evaluated at latitude `phi`.
        
        """
        # Transform to the standard params
        mu, sigma = self.get_standard_params(mean, std)
        var = sigma ** 2
        b = 1 - xi
        return (
            1.0
            / (b * np.sqrt(2 * np.pi * var))
            * np.exp(-((np.log(b) - mu) ** 2) / (2 * var))
        )
