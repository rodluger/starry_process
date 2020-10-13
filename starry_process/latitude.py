from .wigner import R
from .integrals import WignerIntegral
from .transforms import LatitudeTransform
from .ops import LatitudeIntegralOp, CheckBoundsOp


class LatitudeIntegral(WignerIntegral):
    def _ingest(self, a, b, **kwargs):
        """
        Ingest the parameters of the distribution and 
        set up the transform and rotation operators.

        """
        # Ingest
        self._a = CheckBoundsOp(name="a", lower=0, upper=1)(a)
        self._b = CheckBoundsOp(name="b", lower=0, upper=1)(b)
        self._params = [self._a, self._b]

        # Set up the transform
        self._transform = LatitudeTransform(**kwargs)

        # Set up the rotation operator
        self._R = R(
            self._ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1
        )

        # Compute the integrals
        self._integral_op = LatitudeIntegralOp(self._ydeg, **kwargs)
        self._alpha, self._beta = self._transform._ab_to_alphabeta(
            self._a, self._b
        )
        self._q, _, _, self._Q, _, _ = self._integral_op(
            self._alpha, self._beta
        )
