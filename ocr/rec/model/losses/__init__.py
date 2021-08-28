# -*- coding: utf-8 -*-
# @Time : 2021/8/10 下午2:57
# @Author: yl
# @File: __init__.py
__all__ = ['build_loss']


def build_loss(config):
    from .ctc import CTCLoss, CTCLayer
    supported_dict = ['CTCLoss', "CTCLayer"]
    module_type = config.pop('type')
    assert module_type in supported_dict, f"Not supported for {module_type}, Only {supported_dict} are supported."
    module_class = eval(module_type)(**config)
    return module_class
