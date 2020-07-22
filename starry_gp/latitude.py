from .transform import TransformIntegral
from starry._c_ops import LatitudeGP
import numpy as np
from scipy.special import gamma, hyp2f1


class LatitudeIntegral(TransformIntegral):
    def __init__(self, ydeg):
        self._c_op = LatitudeGP(ydeg)
        super().__init__(ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1)
        self.set_params()

    def get_alpha_beta(self, mu, nu):
        """
        Compute the parameters `alpha` and `beta` of the Beta distribution
        in cos(lat), given the mean `mu` and normalized variance `nu`.

        The mean `mu` is the mean of the Beta distribution, valid in (0, 1).
        The normalized variance `nu` is the variance of the Beta distribution
        divided by `mu * (1 - mu)`, valid in `(0, 1)`.

        """
        assert mu > 0 and mu < 1, "mean must be in (0, 1)."
        assert nu > 0 and nu < 1, "variance must be in (0, 1)."
        alpha = mu * (1 / nu - 1)
        beta = (1 - mu) * (1 / nu - 1)
        return alpha, beta

    def _compute_basis_integrals(self, mu_beta=0.5, nu_beta=0.01):
        alpha, beta = self.get_alpha_beta(mu_beta, nu_beta)
        self._c_op.compute(alpha, beta)
        self.q = self._c_op.q
        self.Q = self._c_op.Q
