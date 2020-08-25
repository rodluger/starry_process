import theano.tensor as tt
from theano.tensor.nlinalg import eigh
from theano.ifelse import ifelse


def eigen(Q, neig=None):
    """
    Returns the square root of a hermitian matrix `Q`,
    computed via eigendecomposition, such that:

        eigen(Q) . eigen(Q)^T = Q

    """
    # TODO: compute only `neig` eigenvalues?
    w, U = eigh(Q)
    U = tt.dot(U, tt.diag(tt.sqrt(tt.maximum(0, w))))
    if neig is not None:
        return U[:, -neig:]
    else:
        return U


class MomentIntegral(object):
    def __init__(self, parent=None, ydeg=None, **kwargs):
        self.parent = parent
        if self.parent is not None:
            self.parent.child = self
            self.ydeg = self.parent.ydeg
        else:
            self.ydeg = ydeg
            assert self.ydeg is not None, "please provide `ydeg`."
        self.child = None
        self.N = (self.ydeg + 1) ** 2
        self.n = 2 * self.ydeg + 1
        self.neig = self.N
        self._set = False
        self.e = None
        self.eigE = None
        self._precompute(**kwargs)

    def set_params(self, *args, **kwargs):
        self._set_params(*args, **kwargs)
        self.e = None
        self.eigE = None
        if self.parent is not None:
            self.parent.e = None
            self.parent.eigE = None
        self._set = True

    def first_moment(self):
        assert self._set, "must call `set_params()` first."
        if self.e is None:
            if self.child is None:
                self.e = self._first_moment()
            else:
                self.e = self._first_moment(self.child.first_moment())
        return self.e

    def second_moment(self):
        assert self._set, "must call `set_params()` first."
        if self.eigE is None:
            if self.child is None:
                self.eigE = self._second_moment()
            else:
                self.eigE = self._second_moment(self.child.second_moment())
        return self.eigE

    # All of the following methods must be defined in the subclasses:

    def _precompute(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _set_params(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _first_moment(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _second_moment(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")


class WignerIntegral(MomentIntegral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neig = self.n

    def _compute_basis_integrals(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _compute_U(self):
        self.U = eigen(self.Q, self.neig)

    def _compute_t(self):
        self.t = [None for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            self.t[l] = tt.dot(self.R[l], self.q[l ** 2 : (l + 1) ** 2])

    def _compute_T(self):
        self.T = [None for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            self.T[l] = tt.swapaxes(tt.dot(self.R[l], self.U[i]), 1, 2)

    def _set_params(self, *args, **kwargs):
        self._compute_basis_integrals(*args, **kwargs)
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
        sqrtC = ifelse(sqrtC.shape[1] > self.N, eigen(sqrtC @ sqrtC.T))
        return sqrtC
