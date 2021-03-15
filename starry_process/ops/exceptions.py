# -*- coding: utf-8 -*-
import numpy as np
from ..compat import tt, Op, Apply, floatX

__all__ = ["CheckBoundsOp", "CheckVectorSizeOp"]


class CheckBoundsOp(Op):
    """"""

    __props__ = ("lower", "upper", "name", "tol")

    def __init__(self, lower=-np.inf, upper=np.inf, name=None, tol=1e-6):
        self.lower = lower
        self.upper = upper
        self.tol = tol
        if name is None:
            self.name = "parameter"
        else:
            self.name = name

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(inputs[0]).astype(floatX)]
        outputs = [inputs[0].type()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        return args[-1]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.array(inputs[0])
        if np.any(
            (inputs[0] < self.lower - self.tol)
            | (inputs[0] > self.upper + self.tol)
        ):
            low = np.where((inputs[0] < self.lower - self.tol))[0]
            high = np.where((inputs[0] > self.upper + self.tol))[0]
            if len(low):
                value = np.atleast_1d(inputs[0])[low[0]]
                sign = "<="
                bound = self.lower
            else:
                value = np.atleast_1d(inputs[0])[high[0]]
                sign = ">="
                bound = self.upper
            raise ValueError(
                "%s out of bounds: %f %s %f" % (self.name, value, sign, bound)
            )

    def grad(self, inputs, gradients):
        return [1.0 * gradients[0]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


class CheckVectorSizeOp(Op):
    """"""

    __props__ = ("name", "size")

    def __init__(self, name=None, size=None):
        self.size = size
        if name is None:
            self.name = "vector"
        else:
            self.name = name

    def make_node(self, *inputs):
        inputs = [tt.as_tensor_variable(inputs[0]).astype(floatX)]
        outputs = [inputs[0].type()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        return args[-1]

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.array(inputs[0])
        if inputs[0].size != self.size:
            raise ValueError(
                "Vector `%s` has the wrong size. Expected %d, got %d."
                % (self.name, self.size, inputs[0].size)
            )

    def grad(self, inputs, gradients):
        return [1.0 * gradients[0]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
