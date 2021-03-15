# -*- coding: utf-8 -*-
from ..starry_process_version import __version__
from ..defaults import defaults
from ..compat import theano, tt, COp, Apply, floatX
import sys
import pkg_resources

# Allow C code caching even in dev mode?
try:
    from .. import CACHE_DEV_C_CODE
except:
    CACHE_DEV_C_CODE = False

__all__ = ["BaseOp", "IntegralOp"]


class BaseOp(COp):

    __props__ = ("ydeg", "udeg", "compile_args")
    func_file = None
    func_name = None

    def __init__(
        self,
        ydeg=defaults["ydeg"],
        udeg=defaults["udeg"],
        compile_args=[],
        **kwargs
    ):
        self.ydeg = ydeg
        self.N = (self.ydeg + 1) ** 2
        self.udeg = udeg
        self.NLU = (self.ydeg + self.udeg + 1) ** 2
        assert type(compile_args) is list, "arg `compile_args` must be a list"
        for item in compile_args:
            assert (
                type(item) is tuple
            ), "items in `compile_args` must be tuples"
            assert len(item) == 2, "tuples in `compile_args` must have 2 items"
        self.compile_args = tuple(compile_args)
        super().__init__(self.func_file, self.func_name)

    def perform(self, *args):
        raise NotImplementedError("Only C op is implemented")

    def c_code_cache_version(self, *args, **kwargs):
        if ("dev" in __version__) and not CACHE_DEV_C_CODE:
            return ()
        else:
            v = []
            for sv in __version__.split("."):
                try:
                    v.append(int(sv))
                except:
                    v.append(sv)
            return tuple(v)

    def c_headers(self, *args, **kwargs):
        return [
            "utils.h",
            "special.h",
            "latitude.h",
            "wigner.h",
            "eigh.h",
            "flux.h",
            "theano_helpers.h",
            "vector",
        ]

    def c_header_dirs(self, *args, **kwargs):
        dirs = [
            pkg_resources.resource_filename("starry_process", "ops/include")
        ]
        dirs += [
            pkg_resources.resource_filename(
                "starry_process", "ops/vendor/eigen_3.3.5"
            )
        ]
        return dirs

    def c_compile_args(self, *args, **kwargs):
        args = ["-std=c++14", "-O2", "-DNDEBUG"]
        if sys.platform == "darwin":
            args += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        args += ["-DSP__LMAX={0}".format(self.ydeg)]
        args += ["-DSP__UMAX={0}".format(self.udeg)]
        for (key, value) in self.compile_args:
            if key.startswith("SP_"):
                args += ["-D{0}={1}".format(key, value)]
        return args


class IntegralOp(BaseOp):
    def make_node(self, alpha, beta):
        in_args = [
            tt.as_tensor_variable(arg).astype(floatX) for arg in [alpha, beta]
        ]
        out_args = [
            tt.TensorType(dtype=floatX, broadcastable=[False])(),
            tt.TensorType(dtype=floatX, broadcastable=[False])(),
            tt.TensorType(dtype=floatX, broadcastable=[False])(),
            tt.TensorType(dtype=floatX, broadcastable=[False, False])(),
            tt.TensorType(dtype=floatX, broadcastable=[False, False])(),
            tt.TensorType(dtype=floatX, broadcastable=[False, False])(),
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
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
