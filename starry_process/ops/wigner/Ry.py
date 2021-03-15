# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from ...compat import Apply, theano, tt, floatX

__all__ = ["RyOp"]


class RyOp(BaseOp):
    func_file = "./Ry.cc"
    func_name = "APPLY_SPECIFIC(Ry)"

    def make_node(self, theta):
        in_args = [
            tt.as_tensor_variable(arg).astype(floatX) for arg in [theta]
        ]
        out_args = [
            tt.TensorType(dtype=floatX, broadcastable=[False])(),
            tt.TensorType(dtype=floatX, broadcastable=[False])(),
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        nwig = (
            (self.ydeg + 1) * (2 * self.ydeg + 1) * (2 * self.ydeg + 3)
        ) // 3
        return ([nwig], [nwig])

    def grad(self, inputs, gradients):
        (theta,) = inputs
        Ry, dRydtheta = self(*inputs)
        bRy = gradients[0]
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )
        btheta = tt.sum(bRy * dRydtheta)
        return (btheta,)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
