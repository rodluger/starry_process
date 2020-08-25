# -*- coding: utf-8 -*-

__all__ = ["LatitudeIntegralOp"]

import theano
import theano.tensor as tt
from theano import gof
from .base_op import LatitudeIntegralBaseOp


class LatitudeIntegralOp(LatitudeIntegralBaseOp):

    __props__ = ()
    func_file = "./latitude.cc"
    func_name = "APPLY_SPECIFIC(latitude)"

    def __init__(self, ydeg):
        self.ydeg = ydeg
        self.N = (self.ydeg + 1) ** 2

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
        bf = gradients[0]
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )
        bc = tt.sum(
            tt.reshape(bf, (1, bf.size))
            * tt.reshape(dfdcl, (c.size, bf.size)),
            axis=-1,
        )
        bb = bf * dfdb
        br = bf * dfdr
        return bc, bb, br, tt.zeros_like(los)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
