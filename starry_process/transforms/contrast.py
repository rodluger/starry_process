from .transforms import IdentityTransform
import numpy as np

__all__ = ["ContrastTransform"]


class ContrastTransform(IdentityTransform):
    def pdf(self, x, mu, sigma):
        v = sigma ** 2
        b = (1 - mu) ** 2
        mu_b = np.log(b / np.sqrt(b + v))
        var_b = np.log(1 + v / b) ** 2
        return (
            1.0
            / ((1 - x) * np.sqrt(2 * np.pi * var_b))
            * np.exp(-((np.log(1 - x) - mu_b) ** 2) / (2 * var_b))
        )

    def transform_params(self, mu, sigma):
        # Bounds checks
        mu = np.array(mu)
        sigma = np.array(sigma)
        assert np.all(sigma > 0), "sigma is out of bounds"
        return mu, sigma
