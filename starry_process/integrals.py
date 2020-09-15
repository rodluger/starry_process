from .math import matrix_sqrt
import theano.tensor as tt
from theano.ifelse import ifelse


class MomentIntegral(object):
    def __init__(self, ydeg, child=None, driver="numpy", **kwargs):
        self.ydeg = ydeg
        self.driver = driver
        if child is None:

            class NoChild(object):
                def first_moment(self):
                    return None

                def second_moment(self):
                    return None

            self.child = NoChild()
        else:
            self.child = child
        self.N = (self.ydeg + 1) ** 2
        self.n = 2 * self.ydeg + 1
        self._precompute(**kwargs)
        self._set_params(**kwargs)

    @property
    def neig(self):
        return self.N

    def first_moment(self):
        return self._first_moment(self.child.first_moment())

    def second_moment(self):
        return self._second_moment(self.child.second_moment())

    # All of the following methods must be defined in the subclasses:

    def _precompute(self, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _set_params(self, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _first_moment(self, e):
        raise NotImplementedError("Must be subclassed.")

    def _second_moment(self, eigE):
        raise NotImplementedError("Must be subclassed.")


class WignerIntegral(MomentIntegral):
    @property
    def neig(self):
        return self.n

    def _compute_basis_integrals(self, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _compute_U(self):
        self.U = matrix_sqrt(self.Q, neig=self.neig, driver=self.driver)

    def _compute_t(self):
        self.t = [None for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            self.t[l] = tt.dot(self.R[l], self.q[l ** 2 : (l + 1) ** 2])

    def _compute_T(self):
        self.T = [None for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            self.T[l] = tt.swapaxes(tt.dot(self.R[l], self.U[i]), 1, 2)

    def _set_params(self, **kwargs):
        self._compute_basis_integrals(**kwargs)
        self._compute_U()
        self._compute_t()
        self._compute_T()

    def _first_moment(self, e):
        mu = tt.zeros(self.N)
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            mu = tt.set_subtensor(mu[i], tt.dot(self.t[l], e[i]))
        return mu

    def _second_moment(self, eigE):
        sqrtC = tt.zeros((self.N, self.neig, eigE.shape[-1]))
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            sqrtC = tt.set_subtensor(sqrtC[i], tt.dot(self.T[l], eigE[i]))
        sqrtC = tt.reshape(sqrtC, (self.N, -1))
        # Sometimes it's useful to reduce the size of `sqrtC` by
        # finding the equivalent lower dimension representation
        # via eigendecomposition. This is not an approximation!
        # TODO: Investigate the numerical stability of the gradient
        # of this operation! Many of the eigenvalues are VERY small.
        sqrtC = ifelse(
            sqrtC.shape[1] > self.N,
            matrix_sqrt(tt.dot(sqrtC, sqrtC.T), driver=self.driver),
            sqrtC,
        )
        return sqrtC
