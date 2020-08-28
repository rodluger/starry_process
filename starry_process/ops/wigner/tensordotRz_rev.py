# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from theano import gof
import theano
import theano.tensor as tt

__all__ = ["tensordotRzRevOp"]


class tensordotRzRevOp(BaseOp):
    func_file = "./tensordotRz_rev.cc"
    func_name = "APPLY_SPECIFIC(tensordotRz_rev)"

    def make_node(self, M, theta, bf):
        in_args = []
        dtype = theano.config.floatX
        for a in [M, theta, bf]:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)
        out_args = [
            tt.TensorType(dtype=dtype, broadcastable=[False, False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False,])(),
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        K = shapes[0][0]
        return ([K, self.N], [K,])
