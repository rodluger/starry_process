from starry_process.compat import theano, tt
from theano.configparser import change_flags
from starry_process.ops import RxOp
import numpy as np


def test_Rx_grad(
    ydeg=5, theta=np.pi / 3, abs_tol=1e-5, rel_tol=1e-5, eps=1e-7
):
    with change_flags(compute_test_value="off"):
        op = RxOp(ydeg)
        theano.gradient.verify_grad(
            lambda theta: op(theta)[0],
            (theta,),
            n_tests=1,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            rng=np.random,
        )
