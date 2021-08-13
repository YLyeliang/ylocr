# -*- coding: utf-8 -*-
# @Time : 2021/8/10 下午2:57
# @Author: yl
# @File: __init__.py

__all__ = ['build_backbone']


def build_backbone(config):
    from .resnet import ResNet,ResNetV2
    supported_dict = ['ResNet',"ResNetV2"]
    module_type = config.pop('type')
    assert module_type in supported_dict, f"Not supported for {module_type}, Only {supported_dict} are supported."
    module_class = eval(module_type)(**config)
    return module_class
