# -*- coding: utf-8 -*-
# @Time : 2021/8/6 下午4:59
# @Author: yl
# @File: step.py
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class StepDecoder(keras.Model):
    def __init__(self,
                 num_classes=None,
                 rnn_flag=False):
        super(StepDecoder, self).__init__()
        self.num_classes = num_classes
        self.flatten = layers.TimeDistributed(layers.Flatten(), name="flatten")

        self.drop_out = layers.Dropout(rate=0.2)
        self.dense = layers.TimeDistributed(layers.Dense(num_classes))
        self.softmax = tf.nn.softmax

    def call(self, x, training=None, mask=None):
        x = self.flatten(x)
        if training:
            x = self.drop_out(x)

        x = self.dense(x)
        pred = self.softmax(x)
        return pred
