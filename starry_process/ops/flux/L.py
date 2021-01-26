# -*- coding: utf-8 -*-
from ..base_op import BaseOp
import theano.tensor as tt
from ...compat import Apply

__all__ = ["LOp"]


class LOp(BaseOp):
    func_file = "./L.cc"
    func_name = "APPLY_SPECIFIC(L)"

    def make_node(self, u):
        in_args = [
            tt.as_tensor_variable(arg).astype(tt.config.floatX) for arg in [u]
        ]
        out_args = [
            tt.TensorType(
                dtype=tt.config.floatX, broadcastable=[False, False]
            )()
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        return ([self.NLU, self.N],)
