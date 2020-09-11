from starry_process.math import matrix_sqrt
import numpy as np
from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags
import theano.tensor as tt
from theano.tensor import nlinalg
import pytest


def test_grad():
    with change_flags(compute_test_value="off"):
        Q = np.random.randn(256, 256)
        Q += Q.T
        verify_grad(lambda la: tt.sum(matrix_sqrt(Q)), (Q,), n_tests=1)


def test_grad_low_rank():
    with change_flags(compute_test_value="off"):
        Q = np.random.randn(256, 16)
        Q = Q @ Q.T
        verify_grad(lambda la: tt.sum(matrix_sqrt(Q)), (Q,), n_tests=1)
        verify_grad(
            lambda la: tt.sum(matrix_sqrt(Q, neig=16)), (Q,), n_tests=1
        )
