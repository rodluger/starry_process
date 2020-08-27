# -*- coding: utf-8 -*-
from ..starry_process_version import __version__
from theano import gof
import sys
import pkg_resources

__all__ = ["IntegralBaseOp"]


class IntegralBaseOp(gof.COp):

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
        args = ["-std=c++11", "-O2", "-DNDEBUG"]
        if sys.platform == "darwin":
            args += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        args += ["-DSP__LMAX={0}".format(self.ydeg)]
        for key, value in self.compile_kwargs.items():
            if key.startswith("SP_"):
                args += ["-D{0}={1}".format(key, value)]
        return args
