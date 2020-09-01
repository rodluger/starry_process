# -*- coding: utf-8 -*-
from ..base_op import IntegralOp

__all__ = ["LatitudeIntegralOp"]


class LatitudeIntegralOp(IntegralOp):
    func_file = "./latitude.cc"
    func_name = "APPLY_SPECIFIC(latitude)"
