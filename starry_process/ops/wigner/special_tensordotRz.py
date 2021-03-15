# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from ...compat import Apply, theano, tt, floatX
from .special_tensordotRz_rev import special_tensordotRzRevOp

__all__ = ["special_tensordotRzOp"]


class special_tensordotRzOp(BaseOp):
    func_file = "./special_tensordotRz.cc"
    func_name = "APPLY_SPECIFIC(special_tensordotRz)"

    def __init__(self, *args, **kwargs):
        self.grad_op = special_tensordotRzRevOp(*args, **kwargs)
        super().__init__(*args, **kwargs)

    def make_node(self, T, M, theta):
        in_args = [
            tt.as_tensor_variable(arg).astype(floatX) for arg in [T, M, theta]
        ]
        out_args = [tt.TensorType(dtype=floatX, broadcastable=[False])()]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        K = shapes[2][0]
        return ([K],)

    def grad(self, inputs, gradients):
        return [tt.zeros((self.N, self.N))] + self.grad_op(
            *inputs, gradients[0]
        )

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
