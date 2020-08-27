from .integrals import MomentIntegral
from .transforms import ContrastTransform
import theano.tensor as tt


class ContrastIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        self.transform = ContrastTransform(**kwargs)

    def _set_params(self, mu, sigma):
        mu = tt.as_tensor_variable(mu).astype(tt.config.floatX)
        sigma = tt.as_tensor_variable(sigma).astype(tt.config.floatX)
        self.fac1 = mu
        self.fac2 = tt.sqrt(sigma ** 2 + mu ** 2)

    def _first_moment(self, e):
        return self.fac1 * e

    def _second_moment(self, eigE):
        return self.fac2 * eigE
