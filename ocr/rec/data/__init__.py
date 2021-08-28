# -*- coding: utf-8 -*-
# @Time : 2021/8/26 下午2:14
# @Author: yl
# @File: __init__.py


__all__ = ['build_dataset', "build_dataloader"]


def build_dataset(config):
    from .simple_dataset import SimpleDataset
    supported_dict = ['SimpleDataset', ]
    module_type = config.pop('type')
    assert module_type in supported_dict, f"Not supported for {module_type}, Only {supported_dict} are supported."
    module_class = eval(module_type)(**config)
    return module_class


def build_dataloader(config):
    dataset_cfg = config["dataset"]
    loader = config["loader"]

    dataset = build_dataset(dataset_cfg)
    data_loader = dataset(**loader)
    return data_loader
