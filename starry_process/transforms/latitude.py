from .transforms import BetaTransform
import numpy as np

__all__ = ["LatitudeTransform"]


class LatitudeTransform(BetaTransform):

    _name = "latitude"
    _defaults = {
        "mom_grid_res": 100,
        "max_sigma": 0,
        "ln_alpha_min": -5.0,
        "ln_alpha_max": 5.0,
        "ln_beta_min": -5.0,
        "ln_beta_max": 5.0,
        "sigma_lim_tol": 1.5,
        "poly_order": 10,
    }

    def _f(self, x):
        return np.cos(np.pi / 180.0 * x)

    def _jac(self, x):
        return 0.5 * np.abs(np.sin(np.pi / 180.0 * x) * np.pi / 180.0)

    def _finv(self, f_of_x):
        return np.arccos(f_of_x) * 180.0 / np.pi
