from .transforms import IdentityTransform
import numpy as np
from scipy.stats import lognorm as LogNormal

__all__ = ["ContrastTransform"]


class ContrastTransform(IdentityTransform):
    def pdf(self, x, mu, sigma):
        p = sigma ** 2
        q = (1 - mu) ** 2
        mu_b = np.log(q / np.sqrt(q + p))
        var_b = np.log(1 + p / q)
        b = 1 - x
        return (
            1.0
            / (b * np.sqrt(2 * np.pi * var_b))
            * np.exp(-((np.log(b) - mu_b) ** 2) / (2 * var_b))
        )

    def sample(self, mu, sigma, nsamples=1):
        p = sigma ** 2
        q = (1 - mu) ** 2
        mu_b = np.log(q / np.sqrt(q + p))
        var_b = np.log(1 + p / q)
        b = LogNormal.rvs(scale=np.exp(mu_b), s=np.sqrt(var_b), size=nsamples)
        xi = 1 - b
        return xi

    def transform_params(self, mu, sigma):
        # Bounds checks
        mu = np.array(mu)
        sigma = np.array(sigma)
        assert np.all(sigma > 0), "sigma is out of bounds"
        return mu, sigma
