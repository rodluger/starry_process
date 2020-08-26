from .integrals import MomentIntegral
import theano.tensor as tt


class ContrastIntegral(MomentIntegral):
    def _precompute(self, **kwargs):
        pass

    def _set_params(self, mu, sigma):
        # TODO: Simplify me!
        v = sigma ** 2
        b = (1 - mu) ** 2
        mu = tt.log(b / tt.sqrt(b + v))
        var = tt.log(1 + v / b) ** 2
        self.fac1 = 1 - tt.exp(mu + 0.5 * var)
        self.fac2 = tt.sqrt(
            1 - 2 * tt.exp(mu + 0.5 * var) + tt.exp(2 * mu + 2 * var)
        )

    def _first_moment(self, e):
        return self.fac1 * e

    def _second_moment(self, eigE):
        return self.fac2 * eigE
