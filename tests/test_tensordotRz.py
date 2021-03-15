from theano.configparser import change_flags
from starry_process.compat import theano, tt
from starry_process.ops import tensordotRzOp, special_tensordotRzOp
import numpy as np


def test_tensordotRz_grad(ydeg=2, abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    with change_flags(compute_test_value="off"):
        op = tensordotRzOp(ydeg)
        theta = (
            np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]) * np.pi / 180.0
        )
        M = np.ones((len(theta), (ydeg + 1) ** 2))
        theano.gradient.verify_grad(
            op,
            (M, theta),
            n_tests=1,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            rng=np.random,
        )


def test_special_tensordotRz_grad(
    ydeg=2, abs_tol=1e-5, rel_tol=1e-5, eps=1e-7
):
    np.random.seed(0)
    with change_flags(compute_test_value="off"):
        op = special_tensordotRzOp(ydeg)
        theta = (
            np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]) * np.pi / 180.0
        )
        M = np.random.random(((ydeg + 1) ** 2, (ydeg + 1) ** 2))
        T = np.random.random(((ydeg + 1) ** 2, (ydeg + 1) ** 2))
        theano.gradient.verify_grad(
            lambda M, theta: op(T, M, theta),
            (M, theta),
            n_tests=1,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            rng=np.random,
        )
