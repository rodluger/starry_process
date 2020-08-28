# -*- coding: utf-8 -*-
from ..base_op import BaseOp
from theano import gof
import theano
import theano.tensor as tt

__all__ = ["tensordotRzOp"]


class tensordotRzOp(BaseOp):
    func_file = "./tensordotRz.cc"
    func_name = "APPLY_SPECIFIC(tensordotRz)"

    def make_node(self, M, theta):
        in_args = []
        dtype = theano.config.floatX
        for a in [M, theta]:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)
        out_args = [
            tt.TensorType(dtype=dtype, broadcastable=[False, False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False, False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False, False])(),
        ]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        K = shapes[0][0]
        return ([K, self.N], [self.N, self.N], [K, self.N])

    def grad(self, inputs, gradients):
        (M, theta,) = inputs
        f, dfdM, dfdtheta = self(*inputs)
        bf = gradients[0]
        for i, g in enumerate(gradients[1:]):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )
        # TODO! CHECK!
        bM = bf * dfdM
        btheta = bf * dfdtheta
        return (bM, btheta)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
