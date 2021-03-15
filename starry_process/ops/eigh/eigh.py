from ..base_op import BaseOp
from ...compat import Op, Apply, theano, tt, floatX
import numpy as np
from functools import partial
from six.moves import xrange
import scipy.linalg

__all__ = ["EighOp"]


def _numpy_eigh(x, neig):
    try:
        eigvals, eigvecs = np.linalg.eigh(x)
    except np.linalg.LinAlgError:
        # Fail silently
        return (np.ones(neig) * np.nan, np.ones((x.shape[0], neig)) * np.nan)
    return (
        np.ascontiguousarray(eigvals[-neig:]),
        np.ascontiguousarray(eigvecs[:, -neig:]),
    )


def _scipy_eigh(x, neig):
    N = x.shape[0]
    eigvals, eigvecs = scipy.linalg.eigh(x, subset_by_index=(N - neig, N - 1))
    return (np.ascontiguousarray(eigvals), np.ascontiguousarray(eigvecs))


class EighOp(tt.nlinalg.Eig):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    This is adapted from

        https://github.com/Theano/Theano/blob/
        eb6a4125c4f5617e74b10503afc3f334f17cf545/
        theano/tensor/nlinalg.py#L294

    to

        (1) implement faster and more numerically stable gradients
        (2) optionally `scipy.nlinalg.eigh` instead of `numpy.linalg.eigh`.

    """

    _sciop = staticmethod(_scipy_eigh)
    _numop = staticmethod(_numpy_eigh)
    __props__ = ("neig", "driver", "mindiff")

    def __init__(self, neig=None, driver="numpy", mindiff=1e-15):
        self.mindiff = mindiff
        self.neig = neig
        self.driver = driver
        if driver == "scipy":
            self._op = self._sciop
        elif driver == "numpy":
            self._op = self._numop
        else:
            raise ValueError("invalid driver")
        self._grad_op = EighGrad(mindiff=self.mindiff)

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        assert x.ndim == 2
        w = tt.dvector()
        v = tt.dmatrix()
        return Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, v) = outputs
        N = x.shape[0]
        if self.neig is None:
            neig = N
        else:
            neig = self.neig
        w[0], v[0] = self._op(x, neig)

    def grad(self, inputs, g_outputs):
        (x,) = inputs
        w, v = self(x)
        # Replace gradients wrt disconnected variables with
        # zeros. This is a work-around for issue #1063.
        gw, gv = _zero_disconnected([w, v], g_outputs)
        return [self._grad_op(x, w, v, gw, gv)]

    def infer_shape(self, *args):
        shapes = args[-1]
        N = shapes[0][0]
        if self.neig is None:
            neig = N
        else:
            neig = self.neig
        return [(neig,), (N, neig)]


def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads):
        if isinstance(g.type, theano.gradient.DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l


class EighGrad(BaseOp):
    func_file = "./eigh.cc"
    func_name = "APPLY_SPECIFIC(eigh)"

    def __init__(self, mindiff=1e-15):
        compile_args = [("SP__EIGH_MINDIFF", "{:.5e}".format(mindiff))]
        super().__init__(compile_args=compile_args)

    def make_node(self, x, w, v, gw, gv):
        x = tt.as_tensor_variable(x).astype(floatX)
        w = tt.as_tensor_variable(w).astype(floatX)
        v = tt.as_tensor_variable(v).astype(floatX)
        gw = tt.as_tensor_variable(gw).astype(floatX)
        gv = tt.as_tensor_variable(gv).astype(floatX)
        assert x.ndim == 2
        assert w.ndim == 1
        assert v.ndim == 2
        assert gw.ndim == 1
        assert gv.ndim == 2
        out = tt.dmatrix()
        return Apply(self, [x, w, v, gw, gv], [out])

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[0]]


class EighGradPython(Op):
    """
    Gradient of an eigensystem of a Hermitian matrix.
    """

    __props__ = ("mindiff",)

    def __init__(self, mindiff=1e-15):
        self.tri0 = np.tril
        self.tri1 = partial(np.triu, k=1)
        self.mindiff = mindiff

    def make_node(self, x, w, v, gw, gv):
        x, w, v, gw, gv = map(tt.as_tensor_variable, (x, w, v, gw, gv))
        assert x.ndim == 2
        assert w.ndim == 1
        assert v.ndim == 2
        assert gw.ndim == 1
        assert gv.ndim == 2
        out_dtype = theano.scalar.upcast(
            x.dtype, w.dtype, v.dtype, gw.dtype, gv.dtype
        )
        out = tt.matrix(dtype=out_dtype)
        return Apply(self, [x, w, v, gw, gv], [out])

    def perform(self, node, inputs, outputs):
        """
        Implements the "reverse-mode" gradient for the eigensystem of
        a square matrix.
        """
        x, w, v, W, V = inputs
        N = x.shape[0]
        outer = np.outer
        neig = w.shape[0]

        def G(n):
            # NOTE: If two eigenvalues `w` are the same (or very
            # close to each other), the gradient here is +/- inf.
            # (This is expected, I think.) However, the subsequent sum of
            # infinities can lead to NaNs.
            # The cases in which I've encountered this correspond
            # to eigenvalues that are extremely small, so I've found
            # that I get the correct result for the gradient if I
            # simply zero out their contributions.
            # TODO: Figure out a more rigorous workaround for this!
            divisor = np.select(
                [np.abs(w[n] - w) > self.mindiff],
                [1.0 / (w[n] - w)],
                default=0.0,
            )
            return sum(
                v[:, m] * V.T[n].dot(v[:, m]) * divisor[m]
                for m in xrange(neig)
                if m != n
            )

        g = sum(outer(v[:, n], v[:, n] * W[n] + G(n)) for n in xrange(neig))

        # Numpy's eigh(a, 'L') (eigh(a, 'U')) is a function of tril(a)
        # (triu(a)) only.  This means that partial derivative of
        # eigh(a, 'L') (eigh(a, 'U')) with respect to a[i,j] is zero
        # for i < j (i > j).  At the same time, non-zero components of
        # the gradient must account for the fact that variation of the
        # opposite triangle contributes to variation of two elements
        # of Hermitian (symmetric) matrix. The following line
        # implements the necessary logic.
        out = self.tri0(g) + self.tri1(g).T

        # Make sure we return the right dtype even if NumPy performed
        # upcasting in self.tri0.
        outputs[0][0] = np.asarray(out, dtype=node.outputs[0].dtype)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[0]]
