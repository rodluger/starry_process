# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from ...compat import Apply, theano, tt, floatX
from .tensordotRz_rev import tensordotRzRevOp

__all__ = ["tensordotRzOp"]


class tensordotRzOp(BaseOp):
    func_file = "./tensordotRz.cc"
    func_name = "APPLY_SPECIFIC(tensordotRz)"

    def __init__(self, *args, **kwargs):
        self.grad_op = tensordotRzRevOp(*args, **kwargs)
        super().__init__(*args, **kwargs)

    def make_node(self, M, theta):
        in_args = [
            tt.as_tensor_variable(arg).astype(floatX) for arg in [M, theta]
        ]
        out_args = [
            tt.TensorType(dtype=floatX, broadcastable=[False, False])()
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        K = shapes[0][0]
        return ([K, self.N],)

    def grad(self, inputs, gradients):
        return self.grad_op(*inputs, gradients[0])

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
