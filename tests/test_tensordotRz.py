from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags
from starry_process.ops import tensordotRzOp
import numpy as np


def test_tensordotRz_grad(ydeg=2, abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        op = tensordotRzOp(ydeg)
        theta = (
            np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]) * np.pi / 180.0
        )
        M = np.ones((len(theta), (ydeg + 1) ** 2))
        verify_grad(
            op,
            (M, theta,),
            n_tests=1,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
        )


if __name__ == "__main__":
    test_tensordotRz_grad()
