# -*- coding: utf-8 -*-
# @Time : 2021/8/11 下午2:18
# @Author: yl
# @File: __init__.py

__all__ = ['build_metric']


def build_metric(config):
    from .metrics import RecMetric
    supported_dict = ['RecMetric']
    module_type = config.pop('type')
    assert module_type in supported_dict, f"Not supported for {module_type}, Only {supported_dict} are supported."
    module_class = eval(module_type)(**config)
    return module_class
