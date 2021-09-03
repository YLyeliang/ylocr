# -*- coding: utf-8 -*-
# @Time : 2021/8/26 下午5:08
# @Author: yl
# @File: __init__.py


__all__ = ['build_optimizer']


# lr_schedule = {"Cosine": CosineDecay,
#                'Poly': optimizers.schedules.PolynomialDecay,
#                "Constant": lambda x: x}


def build_optimizer(config):
    from .learning_schedules import Constant, Cosine, Poly
    from .optimizer import SGD, Adam
    all_optimizers = ['SGD', 'Adam']
    module_type = config.pop("type")
    assert module_type in all_optimizers, f"Not supported for {module_type}, Only {all_optimizers} are supported."

    lr_cfg = config.pop("lr")
    lr_type = lr_cfg.pop("type")
    lr = eval(lr_type)(**lr_cfg)
    config['learning_rate'] = lr_cfg['initial_learning_rate']

    module_class = eval(module_type)(**config)
    return module_class, lr
