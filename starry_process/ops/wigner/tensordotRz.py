# -*- coding: utf-8 -*-
from ..base_op import BaseOp

__all__ = ["tensordotRzOp"]


class tensordotRzOp(BaseOp):
    func_file = "./tensordotRz.cc"
    func_name = "APPLY_SPECIFIC(tensordotRz)"

    # TODO!
