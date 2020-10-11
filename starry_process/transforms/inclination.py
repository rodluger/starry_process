from .transforms import IdentityTransform
import numpy as np

__all__ = ["InclinationTransform"]


class InclinationTransform(IdentityTransform):
    def pdf(self, x):
        pdf = np.sin(x * np.pi / 180)
        pdf[(x < 0) | (x > 90)] = 0.0
        return pdf

    def sample(self, nsamples=1):
        return np.arccos(np.random.random(size=nsamples)) * 180 / np.pi
