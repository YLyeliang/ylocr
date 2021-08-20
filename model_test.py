# -*- coding: utf-8 -*-
# @Time : 2021/8/6 上午10:28
# @Author: yl
# @File: model_test.py
from ocr.rec.model.backbones.resnet_backup import ResNet
import tensorflow.keras as keras
import tensorflow as tf

model = ResNet()

input = keras.Input([32, 2, 3], batch_size=8)
# model.build(input_shape=[None, 32, 128, 3])
x = keras.layers.TimeDistributed(keras.layers.Flatten())(input)
print(model.summary())
output = model(input)
debug = 1
