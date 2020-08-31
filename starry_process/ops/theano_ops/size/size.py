# -*- coding: utf-8 -*-
from ..base_op import IntegralOp

__all__ = ["SizeIntegralOp"]


class SizeIntegralOp(IntegralOp):
    func_file = "./size.cc"
    func_name = "APPLY_SPECIFIC(size)"
