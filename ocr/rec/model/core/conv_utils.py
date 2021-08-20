# -*- coding: utf-8 -*-
# @Time : 2021/8/6 上午10:43
# @Author: yl
# @File: conv_utils.py
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization

Activations = {"relu": layers.Activation(tf.nn.relu),
               "leaky relu": layers.Activation(tf.nn.leaky_relu),
               "selu": keras.layers.Activation(tf.nn.selu)}


# def Conv(filters, kernel_size, strides=1, dilation=None, padding='same', initializer=tf.initializers.orthogonal(),
#          kernel_regularizer=None, name=None):
#     if dilation is None:
#         dilation = (1, 1)
#     conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
#                   dilation_rate=dilation, kernel_initializer=initializer,
#                   kernel_regularizer=kernel_regularizer, name=name)
#     return conv
#
#
# def ConvBnAct(filters, kernel_size, strides=1, num_layers=2, drop_out=False,
#               initializer=tf.initializers.orthogonal(), kernel_regularizer=None, name=None):
#     conv = []
#     if drop_out:
#         for i in range(num_layers):
#             conv += [
#                 Conv(filters, kernel_size, strides, initializer=initializer, kernel_regularizer=kernel_regularizer),
#                 tf.keras.layers.BatchNormalization(),
#                 tf.keras.layers.ReLU(),
#                 tf.keras.layers.Dropout(0.5)]
#     else:
#         for i in range(num_layers):
#             conv += [
#                 Conv(filters, kernel_size, strides, initializer=initializer, kernel_regularizer=kernel_regularizer),
#                 tf.keras.layers.BatchNormalization(),
#                 tf.keras.layers.ReLU()]
#     return conv
#
#
# def SeparableConv(filters, size, strides=1, padding='same', depth_multiplier=1,
#                   initializer=tf.initializers.orthogonal()):
#     separable_conv = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=size, strides=strides,
#                                                      padding=padding, depth_multiplier=depth_multiplier,
#                                                      depthwise_initializer=initializer,
#                                                      pointwise_initializer=initializer)
#     return separable_conv
#
#
# class convBlock(tf.keras.Sequential):
#     def __init__(self, filters, kernel_size, strides, act="relu", padding="SAME", name=None):
#         super(convBlock, self).__init__()
#         self.conv = Conv(filters, kernel_size, strides=strides, padding=padding, name=name + "_conv")
#         self.bn = BatchNormalization(epsilon=1.001e-5, name=name + "_bn")
#         self.with_act = True if act else False
#         if act:
#             self.act = Activations[act]
#
#     def call(self, x, training=None, mask=None):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.with_act:
#             x = self.act(x)
#         return x
#
#
# class bottleNeck(tf.keras.Model):
#     expansion = 4
#
#     def __init__(self, filters, kernel_size=3, strides=1, act="relu", shortcut=True, name=None):
#         super(bottleNeck, self).__init__()
#         self.shortcut = shortcut
#         if shortcut:
#             self.short = convBlock(4 * filters, 1, strides, None, name=name + "_0")
#
#         self.conv1 = convBlock(filters, 1, strides, act, name=name + "_1")
#
#         self.conv2 = convBlock(filters, kernel_size, 1, act, name=name + "_2")
#
#         self.conv3 = convBlock(4 * filters, 1, 1, act, name=name + "_3")
#
#     def call(self, x, training=None, mask=None):
#         return layers.add([self.short(x), self.conv3(self.conv2(self.conv1(x)))]) if self.shortcut else \
#             self.conv3(self.conv2(self.conv1(x)))
#
#
# class ResLayer(tf.keras.Sequential):
#     def __init__(self, block, filters, kernel_size, strides, num_blocks, act="relu", name=None):
#         super(ResLayer, self).__init__()
#         self.add(block(filters, strides=strides, kernel_size=kernel_size, act=act, name=name + "_block1"))
#         for i in range(2, num_blocks + 1):
#             self.add(block(filters, shortcut=False, name=name + f"_block{i}"))


# def shortBlock(input_tensor, kernel_size, filters, stage, block, dilation, strides=(2, 2), act="relu"):
#     name = 'res' + str(stage) + block + "_"
#
#     x = ConvBlock(input_tensor, 1, filters, kernel_initializer='he_normal', act=act, name=name + "2a")
#
#     x = ConvBlock(x, 1, filters, kernel_initializer='he_normal', act='relu', name=name + "2b")
#
#     x = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', name=name + "2c_conv")(x)
#     x = layers.BatchNormalization(name=name + "2c_bn")(x)
#
#     shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', name=name + "2c_conv")(input_tensor)
#     shortcut = layers.BatchNormalization(name=name + "s_bn")(shortcut)
#
#     x = layers.add([x, shortcut])
#     x = Activations[act](x)
#     return x


# def BasicNeck(input_tensor, kernel_size, filters, stage, block, dilation, act="relu"):


def BottleNeck(input_tensor, kernel_size, filters, stage, block, dilation=(1, 1), strides=(1, 1), shortcut=True,
               act="relu"):
    name = 'res' + str(stage) + block + "_"

    x = ConvBlock(input_tensor, 1, filters, strides=strides, kernel_initializer='he_normal', act=act, name=name + "2a")

    x = ConvBlock(x, 1, filters, kernel_initializer='he_normal', act='relu', name=name + "2b")

    if shortcut:
        x = ConvBlock(x, kernel_size, filters * 4, kernel_initializer='he_normal', act=None, name=name + '2c')

        shortcut = ConvBlock(input_tensor, 1, filters * 4, strides=strides, act=None, name=name + "1")
    else:
        x = ConvBlock(x, kernel_size, filters * 4, kernel_initializer='he_normal', act=act, name=name + '2c')
        shortcut = input_tensor

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ConvBlock(input_tensor, kernel_size, filters, strides=(1, 1), dilation=(1, 1), padding='same',
              kernel_initializer='glorot_uniform', act='relu', name=""):
    conv_name = name + "_conv"
    bn_name = name + "_bn"
    x = layers.Conv2D(filters, kernel_size, strides, padding=padding, dilation_rate=dilation,
                      kernel_initializer=kernel_initializer, name=conv_name)(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name)(x)
    if act:
        x = layers.Activation('relu')(x)
    return x

# def ResBlock(input_tensor,kernel_size,filters,stage,block)
