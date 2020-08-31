from .math import theano_math, numpy_math


class MomentIntegral(object):
    def __init__(self, parent=None, ydeg=None, **kwargs):
        self._math = (
            theano_math if kwargs.get("use_theano", True) else numpy_math
        )
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
        self._set = False
        self.e = None
        self.eigE = None
        self._precompute(**kwargs)

    @property
    def neig(self):
        return self.N

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
    @property
    def neig(self):
        return self.n

    def _compute_basis_integrals(self, *args, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _compute_U(self):
        self.U = self._math.eigen(self.Q, self.neig)

    def _compute_t(self):
        self.t = [None for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            self.t[l] = self._math.dot(
                self.R[l], self.q[l ** 2 : (l + 1) ** 2]
            )

    def _compute_T(self):
        self.T = [None for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            self.T[l] = self._math.swapaxes(
                self._math.dot(self.R[l], self.U[i]), 1, 2
            )

    def _set_params(self, *args, **kwargs):
        self._compute_basis_integrals(*args, **kwargs)
        self._compute_U()
        self._compute_t()
        self._compute_T()

    def _first_moment(self, e):
        mu = self._math.zeros(self.N)
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            try:
                mu = self._math.set_subtensor(
                    mu[i], self._math.dot(self.t[l], e[i])
                )
            except AttributeError:
                mu[i] = self._math.dot(self.t[l], e[i])
        return mu

    def _second_moment(self, eigE):
        sqrtC = self._math.zeros((self.N, self.neig, eigE.shape[-1]))
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            try:
                sqrtC = self._math.set_subtensor(
                    sqrtC[i], self._math.dot(self.T[l], eigE[i])
                )
            except AttributeError:
                sqrtC[i] = self._math.dot(self.T[l], eigE[i])
        sqrtC = self._math.reshape(sqrtC, (self.N, -1))
        # Sometimes it's useful to reduce the size of `sqrtC` by
        # finding the equivalent lower dimension representation
        # via eigendecomposition. This is not an approximation!
        sqrtC = self._math.ifelse(
            sqrtC.shape[1] > self.N,
            self._math.eigen(self._math.dot(sqrtC, sqrtC.T)),
            sqrtC,
        )
        return sqrtC
