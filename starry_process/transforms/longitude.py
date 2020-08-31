from .transforms import IdentityTransform
import numpy as np

__all__ = ["LongitudeTransform"]


class LongitudeTransform(IdentityTransform):
    def pdf(self, x):
        pdf = np.ones_like(x) * 1.0 / 360.0
        pdf[(x < 0) | (x > 360)] = 0.0
        return pdf

    def draw(self, ndraws=1):
        return 360.0 * np.random.random(size=ndraws)

    def transform_params(self):
        return