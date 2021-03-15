# -*- coding: utf-8 -*-
from aesara_theano_fallback import aesara as theano
import aesara_theano_fallback.tensor as tt
from aesara_theano_fallback import ifelse, USE_AESARA
from aesara_theano_fallback.tensor import slinalg
from aesara_theano_fallback.graph import basic, op, params_type, fg

__all__ = [
    "USE_AESARA",
    "theano",
    "tt",
    "ifelse",
    "slinalg",
    "Apply",
    "COp",
    "Op",
    "Params",
    "ParamsType",
    "Node",
    "RandomStream",
    "random_normal",
    "random_uniform",
    "floatX",
]

# Set double precision
floatX = "float64"

# Compatibility imports
Node = basic.Node
Apply = basic.Apply
Op = op.Op
COp = op.ExternalCOp
Params = params_type.Params
ParamsType = params_type.ParamsType
theano.config.floatX = floatX
theano.config.cast_policy = "numpy+floatX"

if USE_AESARA:

    from aesara.tensor.random.utils import RandomStream

    def random_normal(rng, shape):
        return rng.normal(size=shape)

    def random_uniform(rng, shape):
        return rng.uniform(size=shape)


else:

    try:

        from theano.tensor.random.utils import RandomStream

        def random_normal(rng, shape):
            return rng.normal(size=shape)

        def random_uniform(rng, shape):
            return rng.uniform(size=shape)

    except ImportError:

        from theano.tensor.shared_randomstreams import (
            RandomStreams as RandomStream,
        )

        def random_normal(rng, shape):
            return rng.normal(shape)

        def random_uniform(rng, shape):
            return rng.uniform(shape)
