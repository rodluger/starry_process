# -*- coding: utf-8 -*-

__all__ = [
    "Apply",
    "COp",
    "Op",
    "Params",
    "ParamsType",
    "Node",
    "RandomStream",
    "random_normal",
]

try:
    from theano.graph.basic import Apply, Node
    from theano.graph.op import ExternalCOp as COp
    from theano.graph.op import Op
    from theano.graph.params_type import Params, ParamsType
except ImportError:
    from theano.gof.graph import Apply, Node
    from theano.gof.op import COp, Op
    from theano.gof.params_type import Params, ParamsType

try:
    from theano.tensor.random.utils import RandomStream

    def random_normal(rng, shape):
        return rng.normal(size=shape)


except ImportError:
    from theano.tensor.shared_randomstreams import (
        RandomStreams as RandomStream,
    )

    def random_normal(rng, shape):
        return rng.normal(shape)
