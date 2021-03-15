# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from ...compat import theano, tt, floatX, Apply

__all__ = ["rTA1LOp"]


class rTA1LOp(BaseOp):
    func_file = "./rTA1L.cc"
    func_name = "APPLY_SPECIFIC(rTA1L)"

    def __init__(self, *args, **kwargs):
        self.grad_op = rTA1LRevOp(*args, **kwargs)
        super(rTA1LOp, self).__init__(*args, **kwargs)

    def make_node(self, u):
        in_args = [tt.as_tensor_variable(arg).astype(floatX) for arg in [u]]
        out_args = [tt.TensorType(dtype=floatX, broadcastable=[False])()]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        return ([self.N],)

    def grad(self, inputs, gradients):
        return (self.grad_op(inputs[0], gradients[0]),)


class rTA1LRevOp(BaseOp):
    func_file = "./rTA1L_rev.cc"
    func_name = "APPLY_SPECIFIC(rTA1L_rev)"

    def make_node(self, u, bf):
        in_args = [
            tt.as_tensor_variable(arg).astype(floatX) for arg in [u, bf]
        ]
        out_args = [tt.TensorType(dtype=floatX, broadcastable=[False])()]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        return ([self.udeg],)

    def grad(self, inputs, gradients):
        raise NotImplementedError("No gradient available for this op.")
