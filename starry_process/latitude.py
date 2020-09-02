from .wigner import R
from .integrals import WignerIntegral
from .transforms import LatitudeTransform
from .ops import LatitudeIntegralOp
from .math import cast
from .defaults import defaults


class LatitudeIntegral(WignerIntegral):
    def _precompute(self, **kwargs):
        self.transform = LatitudeTransform(**kwargs)
        self.R = R(
            self.ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )
        self._integral_op = LatitudeIntegralOp(self.ydeg, **kwargs)

    def _compute_basis_integrals(
        self, alpha_l=None, beta_l=None, mu_l=None, sigma_l=None, **kwargs
    ):
        p1 = [alpha_l, beta_l]
        p2 = [mu_l, sigma_l]
        if all([p is None for p in p1 + p2]):
            # No params provided: use defaults
            mu_l = defaults["mu_l"]
            sigma_l = defaults["sigma_l"]
            alpha_l, beta_l = self.transform.transform_params(mu_l, sigma_l)
        elif all([p is not None for p in p1]):
            # User provided `alpha` and `beta`
            pass
        elif all([p is not None for p in p2]):
            # User provided `mu` and `sigma`
            alpha_l, beta_l = self.transform.transform_params(mu_l, sigma_l)
        else:
            raise ValueError("invalid parameter combination")
        alpha_l = cast(alpha_l)
        beta_l = cast(beta_l)
        self.q, _, _, self.Q, _, _ = self._integral_op(alpha_l, beta_l)
