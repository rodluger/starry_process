from .ops import EighOp
from .compat import theano, tt, slinalg, Node, floatX
import numpy as np
import scipy.linalg
from inspect import getmro


__all__ = ["is_tensor", "cho_solve", "cho_factor", "cast", "matrix_sqrt"]


def is_tensor(*objs):
    """Return ``True`` if any of ``objs`` is a ``Theano`` object."""
    for obj in objs:
        for c in getmro(type(obj)):
            if c is Node:
                return True
    return False


class Solve(slinalg.Solve):
    """
    Subclassing to override errors due to NaNs.
    Instead, just set the output to NaN.

    """

    def perform(self, node, inputs, output_storage):
        A, b = inputs
        if np.any(np.isnan(A)) or np.any(np.isnan(b)):
            rval = np.ones_like(b) * np.nan
        else:
            if self.A_structure == "lower_triangular":
                rval = scipy.linalg.solve_triangular(A, b, lower=True)
            elif self.A_structure == "upper_triangular":
                rval = scipy.linalg.solve_triangular(A, b, lower=False)
            else:
                rval = scipy.linalg.solve(A, b)
        output_storage[0][0] = rval

    def L_op(self, inputs, outputs, output_gradients):
        """
        Reverse-mode gradient updates for matrix solve operation c = A \\\ b.

        Symbolic expression for updates taken from [#]_.

        References
        ----------
        .. [#] M. B. Giles, "An extended collection of matrix derivative results
          for forward and reverse mode automatic differentiation",
          http://eprints.maths.ox.ac.uk/1079/

        """
        A, b = inputs
        c = outputs[0]
        c_bar = output_gradients[0]
        trans_map = {
            "lower_triangular": "upper_triangular",
            "upper_triangular": "lower_triangular",
        }
        trans_solve_op = Solve(
            # update A_structure and lower to account for a transpose operation
            A_structure=trans_map.get(self.A_structure, self.A_structure),
            lower=not self.lower,
        )
        b_bar = trans_solve_op(A.T, c_bar)
        # force outer product if vector second input
        A_bar = -tt.outer(b_bar, c) if c.ndim == 1 else -b_bar.dot(c.T)
        if self.A_structure == "lower_triangular":
            A_bar = tt.tril(A_bar)
        elif self.A_structure == "upper_triangular":
            A_bar = tt.triu(A_bar)
        return [A_bar, b_bar]


class Cholesky(slinalg.Cholesky):
    """
    Subclassing to override errors due to NaNs.
    Instead, just set the output to NaN.

    """

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        try:
            z[0] = scipy.linalg.cholesky(x, lower=self.lower).astype(x.dtype)
        except (scipy.linalg.LinAlgError, ValueError):
            if self.on_error == "raise":
                raise
            else:
                z[0] = (np.zeros(x.shape) * np.nan).astype(x.dtype)


cho_factor = Cholesky(on_error="nan")


def cho_solve(cho_A, b):
    solve_lower = Solve(A_structure="lower_triangular", lower=True)
    solve_upper = Solve(A_structure="upper_triangular", lower=False)
    return solve_upper(tt.transpose(cho_A), solve_lower(cho_A, b))


def cast(*args, vectorize=False):
    if vectorize:
        if len(args) == 1:
            return tt.reshape(
                tt.as_tensor_variable(args[0]).astype(floatX), (-1,)
            )
        else:
            return [
                tt.reshape(tt.as_tensor_variable(arg).astype(floatX), (-1,))
                for arg in args
            ]
    else:
        if len(args) == 1:
            return tt.as_tensor_variable(args[0]).astype(floatX)
        else:
            return [tt.as_tensor_variable(arg).astype(floatX) for arg in args]


def matrix_sqrt(Q, neig=None, driver="numpy", mindiff=1e-15):
    """
    Returns the matrix square root of `Q`,
    computed via (hermitian) eigendecomposition:

        matrix_sqrt(Q) . matrix_sqrt(Q)^T = Q

    """
    # Eigendecomposition: eigenvalues `w` and eigenvectors `U`
    eigh = EighOp(neig=neig, driver=driver, mindiff=mindiff)
    w, U = eigh(Q)

    # Get the square root of the positive eigenvalues
    sqrtw = tt.switch(
        tt.gt(w, mindiff * tt.ones_like(w)), tt.sqrt(w), tt.zeros_like(w)
    )

    # Dot them in: the result is the matrix square root of `Q`
    return tt.dot(U, tt.diag(sqrtw))
