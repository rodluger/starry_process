from starry_process.math import matrix_sqrt
import numpy as np
from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags
import theano.tensor as tt
from theano.tensor import nlinalg
import pytest


def test_grad():
    with change_flags(compute_test_value="off"):
        np.random.seed(0)
        Q = np.random.randn(10, 10)
        Q = Q @ Q.T
        verify_grad(lambda x: tt.sum(matrix_sqrt(x)), (Q,), n_tests=1)


def test_grad_low_rank():
    with change_flags(compute_test_value="off"):
        np.random.seed(0)
        Q = np.random.randn(3, 2)
        Q = Q @ Q.T
        # verify_grad(lambda x: tt.sum(matrix_sqrt(x)), (Q,), n_tests=1)
        verify_grad(
            lambda x: tt.sum(matrix_sqrt(x, neig=2)), (Q,), n_tests=1,
        )
