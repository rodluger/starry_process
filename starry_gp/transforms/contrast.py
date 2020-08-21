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
        Identity transform: our PDF is a function of mean and std already.
        
        """
        # Bounds checks
        mean = np.array(mean)
        std = np.array(std)
        assert np.all((std > 0)), "std is out of bounds"

        # Transform: identity!
        return mean, std

    def pdf(self, xi, mean, std):
        """
        Return the probability density function evaluated at latitude `phi`.
        
        """
        # Transform to the standard params
        mean, std = self.get_standard_params(mean, std)
        var = std ** 2

        return (
            1.0
            / ((1 - xi) * np.sqrt(2 * np.pi * var))
            * np.exp(-((np.log(1 - xi) - mean) ** 2) / (2 * var))
        )
