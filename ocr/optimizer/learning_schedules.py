# -*- coding: utf-8 -*-
# @Time : 2021/9/1 下午2:38
# @Author: yl
# @File: learning_schedules.py

from tensorflow.keras import optimizers

try:
    from tensorflow.keras.optimizers.schedules import CosineDecay
except:  # tensorflow <=2.3
    from tensorflow.keras.experimental import CosineDecay

Cosine = CosineDecay
Poly = optimizers.schedules.PolynomialDecay

class Constant(object):

    def __init__(self, initial_learning_rate=0.001):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.initial_learning_rate
