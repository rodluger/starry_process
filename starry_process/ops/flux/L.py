# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from ...compat import theano, tt, floatX, Apply

__all__ = ["LOp"]


class LOp(BaseOp):
    func_file = "./L.cc"
    func_name = "APPLY_SPECIFIC(L)"

    def make_node(self, u):
        in_args = [tt.as_tensor_variable(arg).astype(floatX) for arg in [u]]
        out_args = [
            tt.TensorType(dtype=floatX, broadcastable=[False, False])()
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        return ([self.NLU, self.N],)
