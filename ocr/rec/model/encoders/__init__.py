# -*- coding: utf-8 -*-
# @Time : 2021/8/10 下午3:12
# @Author: yl
# @File: __init__.py
__all__ = ['build_encoder']


def build_encoder(config):
    raise NotImplemented
    # from .ctc import CTCLayer
    # supported_dict = ['StepDecoder', 'SequenceDecoder']
    # module_type = config.pop('type')
    # assert module_type in supported_dict, f"Not supported for {module_type}, Only {supported_dict} are supported."
    # module_class = eval(module_type)(**config)
    # return module_class
