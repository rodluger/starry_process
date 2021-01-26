# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from ...compat import Apply
import theano.tensor as tt

__all__ = ["pTA1Op"]


class pTA1Op(BaseOp):
    func_file = "./pTA1.cc"
    func_name = "APPLY_SPECIFIC(pTA1)"

    def make_node(self, x, y, z):
        in_args = [
            tt.as_tensor_variable(x).astype(tt.config.floatX),
            tt.as_tensor_variable(y).astype(tt.config.floatX),
            tt.as_tensor_variable(z).astype(tt.config.floatX),
        ]
        out_args = [
            tt.TensorType(
                dtype=tt.config.floatX, broadcastable=[False, False]
            )()
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        return ([shapes[0][0], self.N],)

    def grad(self, inputs, gradients):
        raise NotImplementedError("No gradient available for this op.")
