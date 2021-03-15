from .math import matrix_sqrt
from .compat import tt, ifelse, Op, Apply, floatX
from .defaults import defaults
import numpy as np


class PDFOp(Op):
    def __init__(self, func):
        self.func = func

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i).astype(floatX) for i in inputs]
        outputs = [tt.TensorType(floatX, (False,))()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs)


class SampleOp(Op):
    def __init__(self, func, nsamples=1):
        self.func = func
        self.nsamples = nsamples

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(i).astype(floatX) for i in inputs]
        outputs = [tt.TensorType(floatX, (False,))()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        return [(self.nsamples,)]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = self.func(*inputs, nsamples=self.nsamples)


class NoChild(object):
    def first_moment(self):
        return None

    def second_moment(self):
        return None


class MomentIntegral(object):
    def __init__(
        self,
        *params,
        ydeg=defaults["ydeg"],
        child=NoChild(),
        driver=defaults["driver"],
        **kwargs
    ):
        self._ydeg = ydeg
        self._driver = driver
        self._child = child
        self._nylm = (self._ydeg + 1) ** 2
        self._angle_fac = np.pi / 180
        self._ingest(*params, **kwargs)
        self._compute()

    def first_moment(self):
        return self._first_moment(self._child.first_moment())

    def second_moment(self):
        return self._second_moment(self._child.second_moment())

    def sample(self, nsamples=1):
        if not hasattr(self, "_sample"):
            raise NotImplementedError("Cannot sample from this distribution.")
        else:
            return SampleOp(self._sample, nsamples=nsamples)(*self._params)

    def pdf(self, x):
        if not hasattr(self, "_pdf"):
            raise NotImplementedError("PDF undefined for this distribution.")
        else:
            return PDFOp(self._pdf)(x, *self._params)

    def log_jac(self):
        if not hasattr(self, "_log_jac"):
            raise NotImplementedError(
                "Jacobian undefined for this distribution."
            )
        else:
            return self._log_jac()

    @property
    def _neig(self):
        return self._nylm

    # All of the following methods must be defined in the subclasses:

    def _ingest(self, *params, **kwargs):
        raise NotImplementedError("Must be subclassed.")

    def _compute(self):
        raise NotImplementedError("Must be subclassed.")

    def _first_moment(self, e):
        raise NotImplementedError("Must be subclassed.")

    def _second_moment(self, eigE):
        raise NotImplementedError("Must be subclassed.")


class WignerIntegral(MomentIntegral):
    @property
    def _neig(self):
        return 2 * self._ydeg + 1

    def _compute(self):
        self._U = matrix_sqrt(self._Q, neig=self._neig, driver=self._driver)
        self._t = [None for l in range(self._ydeg + 1)]
        for l in range(self._ydeg + 1):
            self._t[l] = tt.dot(self._R[l], self._q[l ** 2 : (l + 1) ** 2])
        self._T = [None for l in range(self._ydeg + 1)]
        for l in range(self._ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            self._T[l] = tt.swapaxes(tt.dot(self._R[l], self._U[i]), 1, 2)

    def _first_moment(self, e):
        mu = tt.zeros(self._nylm)
        for l in range(self._ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            mu = tt.set_subtensor(mu[i], tt.dot(self._t[l], e[i]))
        return mu

    def _second_moment(self, eigE):
        sqrtC = tt.zeros((self._nylm, self._neig, eigE.shape[-1]))
        for l in range(self._ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            sqrtC = tt.set_subtensor(sqrtC[i], tt.dot(self._T[l], eigE[i]))
        sqrtC = tt.reshape(sqrtC, (self._nylm, -1))
        # Sometimes it's useful to reduce the size of `sqrtC` by
        # finding the equivalent lower dimension representation
        # via eigendecomposition. This is not an approximation!
        # TODO: Investigate the numerical stability of the gradient
        # of this operation! Many of the eigenvalues are VERY small.
        sqrtC = ifelse(
            sqrtC.shape[1] > self._nylm,
            matrix_sqrt(
                tt.dot(sqrtC, tt.transpose(sqrtC)), driver=self._driver
            ),
            sqrtC,
        )
        return sqrtC

    # All of the following methods must be defined in the subclasses:

    def _ingest(self, *params, **kwargs):
        raise NotImplementedError("Must be subclassed.")
