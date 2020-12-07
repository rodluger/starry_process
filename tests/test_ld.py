from starry_process.ops import rTA1LOp
import numpy as np
from verify_grad import verify_grad
from theano.configparser import change_flags
import theano
import theano.tensor as tt
import pytest


def test_rTA1L_grad():
    with change_flags(compute_test_value="off"):
        op = rTA1LOp(ydeg=1, udeg=3)
        verify_grad(
            lambda u1, u2, u3: op([u1, u2, u3]), [0.5, 0.25, 0.1], n_tests=1,
        )
