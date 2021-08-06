# -*- coding: utf-8 -*-
# @Time : 2021/8/6 上午10:28
# @Author: yl
# @File: model_test.py
from ocr.rec.model.backbone.resnet import ResNet
import tensorflow.keras as keras
import tensorflow as tf

model = ResNet()

input = keras.Input([32, 128, 3], batch_size=8)
model.build(input_shape=[None, 32, 128, 3])
print(model.summary())
output = model(input)
debug = 1
