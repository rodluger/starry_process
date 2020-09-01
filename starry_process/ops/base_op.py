# -*- coding: utf-8 -*-
from ..starry_process_version import __version__
from theano import gof
import theano
import theano.tensor as tt
import sys
import pkg_resources

__all__ = ["BaseOp", "IntegralOp"]


class BaseOp(gof.COp):

    __props__ = ()
    func_file = None
    func_name = None

    def __init__(self, ydeg, **compile_kwargs):
        self.ydeg = ydeg
        self.N = (self.ydeg + 1) ** 2
        self.compile_kwargs = compile_kwargs
        super().__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        if "dev" in __version__:
            return ()
        return tuple(map(int, __version__.split(".")))

    def c_headers(self, compiler):
        return [
            "utils.h",
            "special.h",
            "latitude.h",
            "size.h",
            "wigner.h",
            "flux.h",
            "theano_helpers.h",
            "vector",
        ]

    def c_header_dirs(self, compiler):
        dirs = [
            pkg_resources.resource_filename("starry_process", "ops/include")
        ]
        dirs += [
            pkg_resources.resource_filename(
                "starry_process", "ops/vendor/eigen_3.3.5"
            )
        ]
        return dirs

    def c_compile_args(self, compiler):
        args = ["-std=c++14", "-O2", "-DNDEBUG"]
        if sys.platform == "darwin":
            args += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        args += ["-DSP__LMAX={0}".format(self.ydeg)]
        for key, value in self.compile_kwargs.items():
            if key.startswith("SP_"):
                args += ["-D{0}={1}".format(key, value)]
        return args


class IntegralOp(BaseOp):
    def make_node(self, alpha, beta):
        in_args = [
            tt.as_tensor_variable(arg).astype(tt.config.floatX)
            for arg in [alpha, beta]
        ]
        out_args = [
            tt.TensorType(dtype=tt.config.floatX, broadcastable=[False])(),
            tt.TensorType(dtype=tt.config.floatX, broadcastable=[False])(),
            tt.TensorType(dtype=tt.config.floatX, broadcastable=[False])(),
            tt.TensorType(
                dtype=tt.config.floatX, broadcastable=[False, False]
            )(),
            tt.TensorType(
                dtype=tt.config.floatX, broadcastable=[False, False]
            )(),
            tt.TensorType(
                dtype=tt.config.floatX, broadcastable=[False, False]
            )(),
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
        # Derivs of derivs not implemented
        for i, g in enumerate(list(gradients[1:3]) + list(gradients[4:6])):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                raise ValueError(
                    "can't propagate gradients wrt parameter {0}".format(i + 1)
                )

        # Chain rule
        ba = 0.0
        bb = 0.0
        for g, fa, fb in zip([bq, bQ], [dqda, dQda], [dqdb, dQdb]):
            if not isinstance(g.type, theano.gradient.DisconnectedType):
                ba += tt.sum(g * fa)
                bb += tt.sum(g * fb)

        return ba, bb

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
