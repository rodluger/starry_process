# -*- coding: utf-8 -*-

__all__ = ["SizeIntegralOp"]

import theano
import theano.tensor as tt
from theano import gof
from .base_op import SizeIntegralBaseOp


class SizeIntegralOp(SizeIntegralBaseOp):

    __props__ = ()
    func_file = "./size.cc"
    func_name = "APPLY_SPECIFIC(size)"

    def make_node(self, alpha, beta):
        in_args = []
        dtype = theano.config.floatX
        for a in [alpha, beta]:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)
        out_args = [
            tt.TensorType(dtype=dtype, broadcastable=[False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False, False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False, False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False, False])(),
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return (
            [self.N],
            [self.N],
            [self.N],
            [self.N, self.N],
            [self.N, self.N],
            [self.N, self.N],
        )

    def grad(self, inputs, gradients):
        alpha, beta = inputs
        q, dqda, dqdb, Q, dQda, dQdb = self(*inputs)
        bq = gradients[0]
        bQ = gradients[3]
        for i, g in enumerate(list(gradients[1:3]) + list(gradients[4:6])):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )
        ba = tt.sum(bq * dqda) + tt.sum(bQ * dQda)  # TODO
        bb = tt.sum(bq * dqdb) + tt.sum(bQ * dQdb)  # TODO
        return ba, bb

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
