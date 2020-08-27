# -*- coding: utf-8 -*-
from theano import gof
import sys
import pkg_resources
from ...starry_process_version import __version__

__all__ = ["SizeIntegralBaseOp"]


class SizeIntegralBaseOp(gof.COp):

    __props__ = ()
    func_file = None
    func_name = None

    def __init__(self, ydeg, c=[0.5, 1.0, 0.0, 0.0]):
        self.ydeg = ydeg
        self.c = c
        self.N = (self.ydeg + 1) ** 2
        super().__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        if "dev" in __version__:
            return ()
        return tuple(map(int, __version__.split(".")))

    def c_headers(self, compiler):
        return [
            "utils.h",
            "special.h",
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
        args += ["-DSP_LMAX={0}".format(self.ydeg)]
        for n in range(4):
            args += ["-DSP_C{:d}={:.15f}".format(n, self.c[n])]
        return args
