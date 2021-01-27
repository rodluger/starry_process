# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from ...compat import Apply
import theano.tensor as tt

__all__ = ["special_tensordotRzRevOp"]


class special_tensordotRzRevOp(BaseOp):
    func_file = "./special_tensordotRz_rev.cc"
    func_name = "APPLY_SPECIFIC(special_tensordotRz_rev)"

    def make_node(self, T, M, theta, bf):
        in_args = [
            tt.as_tensor_variable(arg).astype(tt.config.floatX)
            for arg in [T, M, theta, bf]
        ]
        out_args = [
            tt.TensorType(
                dtype=tt.config.floatX, broadcastable=[False, False]
            )(),
            tt.TensorType(
                dtype=tt.config.floatX,
                broadcastable=[
                    False,
                ],
            )(),
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        K = shapes[2][0]
        return (
            [self.N, self.N],
            [
                K,
            ],
        )
