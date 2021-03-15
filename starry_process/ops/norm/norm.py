# -*- coding: utf-8 -*-
import numpy as np
from ...compat import theano, tt, Op, Apply, floatX

__all__ = ["AlphaBetaOp"]


class AlphaBetaOp(Op):
    """"""

    __props__ = ("N",)

    def __init__(self, N=20):
        self.N = N

    def make_node(self, z):
        inputs = [tt.as_tensor_variable(z).astype(floatX)]
        outputs = [
            tt.TensorType(dtype=floatX, broadcastable=[])() for n in range(4)
        ]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        return [(), (), (), ()]

    def perform(self, node, inputs, outputs):
        z = inputs[0]
        fac = 1.0
        alpha = 0.0
        beta = 0.0
        dadz = 0.0
        dbdz = 0.0
        dfdz = 0.0
        for n in range(0, self.N + 1):
            dadz += dfdz
            dbdz += 2 * n * dfdz
            dfdz = (2 * n + 3) * (dfdz * z + fac)
            alpha += fac
            beta += 2 * n * fac
            fac *= z * (2 * n + 3)
        outputs[0][0] = np.array(alpha)
        outputs[1][0] = np.array(beta)
        outputs[2][0] = np.array(dadz)
        outputs[3][0] = np.array(dbdz)

    def grad(self, inputs, gradients):
        z = inputs[0]
        ba = gradients[0]
        bb = gradients[1]
        a, b, dadz, dbdz = self(*inputs)

        # Derivs of derivs not implemented
        for i, g in enumerate(list(gradients[2:])):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )

        bz = tt.as_tensor_variable(0.0)
        for dxdz, bx in zip([dadz, dbdz], [ba, bb]):
            if not isinstance(bx.type, theano.gradient.DisconnectedType):
                bz += dxdz * bx

        return [bz]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
