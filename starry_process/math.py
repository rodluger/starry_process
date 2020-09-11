import theano
import theano.tensor as tt
from theano.tensor import slinalg
from theano.tensor.nlinalg import Eig
from theano.ifelse import ifelse
import numpy as np
from inspect import getmro
from functools import partial
from theano.gof import Op, Apply
from six.moves import xrange
import scipy.linalg

__all__ = ["is_tensor", "cho_solve", "cho_factor", "cast", "matrix_sqrt"]


# Force double precision in Theano
tt.config.floatX = "float64"
tt.config.cast_policy = "numpy+floatX"


def is_tensor(*objs):
    """Return ``True`` if any of ``objs`` is a ``Theano`` object."""
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


def cho_solve(cho_A, b):
    solve_lower = slinalg.Solve(A_structure="lower_triangular", lower=True)
    solve_upper = slinalg.Solve(A_structure="upper_triangular", lower=False)
    return ifelse(
        tt.any(tt.isnan(cho_A)),
        tt.ones_like(b),
        solve_upper(tt.transpose(cho_A), solve_lower(cho_A, b)),
    )


def cho_factor(A):
    cholesky = slinalg.Cholesky(on_error="nan")
    return ifelse(tt.any(tt.isnan(A)), tt.ones_like(A) * np.nan, cholesky(A))


def cast(*args):
    if len(args) == 1:
        return tt.as_tensor_variable(args[0]).astype(tt.config.floatX)
    else:
        return [
            tt.as_tensor_variable(arg).astype(tt.config.floatX) for arg in args
        ]


def _numpy_eigh(x, neig):
    eigvals, eigvecs = np.linalg.eigh(x)
    return eigvals[-neig:], eigvecs[:, -neig:]


def _scipy_eigh(x, neig):
    N = x.shape[0]
    return scipy.linalg.eigh(x, subset_by_index=(N - neig, N - 1))


class Eigh(Eig):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    This is adapted from

        https://github.com/Theano/Theano/blob/
        eb6a4125c4f5617e74b10503afc3f334f17cf545/
        theano/tensor/nlinalg.py#L294

    to use `scipy.nlinalg.eigh` instead of `numpy.linalg.eigh`.

    """

    _sciop = staticmethod(_scipy_eigh)
    _numop = staticmethod(_numpy_eigh)
    __props__ = ("mindiff",)

    def __init__(self, neig=None, driver="numpy", mindiff=1e-15):
        self.mindiff = mindiff
        self.neig = neig
        if driver == "scipy":
            self._op = self._sciop
        elif driver == "numpy":
            self._op = self._numop
        else:
            raise ValueError("invalid driver")

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
        return [EighGrad(self.neig, self.mindiff)(x, w, v, gw, gv)]

    def infer_shape(self, node, shapes):
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


class EighGrad(Op):
    """
    Gradient of an eigensystem of a Hermitian matrix.

    """

    __props__ = ("mindiff",)

    def __init__(self, neig=None, mindiff=1e-15):
        self.neig = neig
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
        if self.neig is None:
            neig = N
        else:
            neig = self.neig

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

    def infer_shape(self, node, shapes):
        return [shapes[0]]


def eigh(a, neig=None, driver="numpy", mindiff=1e-15):
    return Eigh(neig=neig, driver=driver, mindiff=mindiff)(a)


def matrix_sqrt(Q, neig=None, driver="numpy", mindiff=1e-15):
    """
    Returns the matrix square root of `Q`,
    computed via (hermitian) eigendecomposition:

        matrix_sqrt(Q) . matrix_sqrt(Q)^T = Q

    """
    # Eigendecomposition: eigenvalues `w` and eigenvectors `U`
    w, U = eigh(Q, neig=neig, driver=driver, mindiff=mindiff)

    # Get the square root of the positive eigenvalues
    sqrtw = tt.switch(tt.gt(w, tt.zeros_like(w)), tt.sqrt(w), tt.zeros_like(w))

    # Dot them in: the result is the matrix square root of `Q`
    return tt.dot(U, tt.diag(sqrtw))
