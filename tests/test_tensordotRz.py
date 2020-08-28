from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags
from starry_process.ops import tensordotRzOp
import numpy as np


def test_tensordotRz():

    ydeg = 2
    K = 3

    op = tensordotRzOp(ydeg)

    theta = np.linspace(-np.pi, np.pi, K)
    M = np.ones((K, (ydeg + 1) ** 2))
    f = op(M, theta)[0].eval()
    dfdM = op(M, theta)[1].eval()
    dfdtheta = op(M, theta)[2].eval()

    print(f)
    print(dfdM)
    print(dfdtheta)


if __name__ == "__main__":
    test_tensordotRz()
