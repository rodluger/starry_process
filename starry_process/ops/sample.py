# -*- coding: utf-8 -*-
import numpy as np
from ..compat import tt, Op, Apply, floatX

__all__ = ["SampleYlmTemporalOp"]


class SampleYlmTemporalOp(Op):
    """"""

    def make_node(self, *inputs):
        inputs = [
            tt.as_tensor_variable(inputs[0]).astype(floatX),
            tt.as_tensor_variable(inputs[1]).astype(floatX),
            tt.as_tensor_variable(inputs[2]).astype(floatX),
        ]
        outputs = [inputs[2].type()]
        return Apply(self, inputs, outputs)

    def infer_shape(self, *args):
        shapes = args[-1]
        return [shapes[-1]]

    def perform(self, node, inputs, outputs):
        # TODO: Implement this in C++
        Ly, Lt, U = inputs
        nsamples, Nt, Ny = U.shape
        y = np.zeros((nsamples, Nt, Ny))
        for i in range(nsamples):
            for j in range(Ny):
                for k in range(Nt):
                    y[i, k] += Ly.T[j] * (Lt[k] @ U[i, :, j])
        outputs[0][0] = y
