# -*- coding: utf-8 -*-
# @Time : 2021/8/26 下午5:08
# @Author: yl
# @File: __init__.py


__all__ = ['build_optimizer']

from tensorflow.keras import optimizers

try:
    from tensorflow.keras.optimizers.schedules import CosineDecay
except:  # tensorflow <=2.3
    from tensorflow.keras.experimental import CosineDecay

all_optimizers = {"SGD": optimizers.SGD,
                  "Adam": optimizers.Adam}
Adam = optimizers.Adam
SGD = optimizers.SGD
Cosine = CosineDecay
Poly = optimizers.schedules.PolynomialDecay
Constant = lambda x: x


# lr_schedule = {"Cosine": CosineDecay,
#                'Poly': optimizers.schedules.PolynomialDecay,
#                "Constant": lambda x: x}


def build_optimizer(config):
    module_type = config.pop("type")
    assert module_type in all_optimizers.keys(), f"Not supported for {module_type}, Only {all_optimizers.keys()} are supported."

    lr_cfg = config.pop("lr")
    lr_type = lr_cfg.pop("type")
    lr = eval(lr_type)(**lr_cfg)
    config['learning_rate'] = lr

    module_class = eval(module_type)(**config)
    return module_class
