# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from ...compat import Apply, theano, tt, floatX

__all__ = ["rTA1Op"]


class rTA1Op(BaseOp):
    func_file = "./rTA1.cc"
    func_name = "APPLY_SPECIFIC(rTA1)"

    def make_node(self):
        in_args = []
        out_args = [tt.TensorType(dtype=floatX, broadcastable=[False])()]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        return ([self.N],)

    def grad(self, inputs, gradients):
        raise NotImplementedError("No gradient available for this op.")
