# -*- coding: utf-8 -*-
# @Time : 2021/8/18 上午8:52
# @Author: yl
# @File: resnet.py
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from ..core.conv_utils import BottleNeck, ConvBlock, BasicBlock


class ResNet(object):
    arch_settings = {
        # 18: (BasicBlock, (2, 2, 2, 2)),
        # 34: (BasicBlock, (3, 4, 6, 3)),
        50: (BottleNeck, (3, 4, 6, 3)),
        101: (BottleNeck, (3, 4, 23, 3)),
        152: (BottleNeck, (3, 8, 36, 3))

    }

    def __init__(self, depth=50,
                 stem_channels=64,
                 strides=((1, 1), (2, 2), (2, 1), (2, 1)),
                 dilations=(1, 1, 1, 1),
                 act='relu'):
        super(ResNet, self).__init__()

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks

        self.strides = strides
        self.dilations = dilations
        self.act = act
        self.filters = stem_channels

        self.stem = self.make_stem(stem_channels)
        self.inplanes = stem_channels

    def make_stem(self, stem_channels):
        def stem(input_tensor):
            x = ConvBlock(input_tensor, 7, stem_channels, strides=(2, 2), kernel_initializer='he_normal')
            x = layers.MaxPool2D(3, strides=(2, 2), padding='same')(x)
            return x

        return stem

    def __call__(self, x):
        x = self.stem(x)
        # input_tensor, kernel_size, filters, stage, block, dilation, strides=(1, 1), shortcut=True, act="relu"
        for l, strides in enumerate(self.strides):
            x = self.block(x, 3, self.filters, stage=l + 2, block='a', dilation=self.dilations[l], strides=strides,
                           shortcut=True)
            for i in range(self.stage_blocks[l] - 1):
                x = self.block(x, 3, self.filters, stage=l + 2, block=chr(ord('b') + i), dilation=self.dilations[i],
                               shortcut=False, act=self.act)
        return x


class ResNetV2(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (BottleNeck, (3, 4, 6, 3)),
        101: (BottleNeck, (3, 4, 23, 3)),
        152: (BottleNeck, (3, 8, 36, 3))

    }

    def __init__(self, depth=50,
                 stem_channels=64,
                 strides=((1, 1), (2, 1), (2, 1), (2, 1)),
                 dilations=(1, 1, 1, 1),
                 max_pool_first=True,
                 max_pool_last=True,
                 act='relu'):
        super(ResNetV2, self).__init__(depth=depth, stem_channels=stem_channels, strides=strides, dilations=dilations,
                                       act=act)

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks

        self.strides = strides
        self.dilations = dilations
        self.act = act
        self.filters = stem_channels

        self.stem = self.make_stem(stem_channels)
        self.inplanes = stem_channels
        self.max_pool_first = max_pool_first
        self.max_pool_last = max_pool_last
        if max_pool_first:
            self.max_pool1 = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        if max_pool_last:
            self.max_pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')

    def make_stem(self, stem_channels):
        def stem(input_tensor):
            x = ConvBlock(input_tensor, 3, stem_channels, strides=1, kernel_initializer="he_normal", act='relu',
                          name='1')
            x = ConvBlock(x, 3, stem_channels, strides=1, kernel_initializer='he_normal', act='relu', name='2')
            return x

        return stem

    def __call__(self, x):
        x = self.stem(x)
        if self.max_pool_first:
            x = self.max_pool1(x)
        # input_tensor, kernel_size, filters, stage, block, dilation, strides=(1, 1), shortcut=True, act="relu"
        for l, strides in enumerate(self.strides):
            x = self.block(x, 3, self.filters, stage=l + 2, block='a', dilation=self.dilations[l], strides=strides,
                           shortcut=True)
            for i in range(self.stage_blocks[l] - 1):
                x = self.block(x, 3, self.filters, stage=l + 2, block=chr(ord('b') + i), dilation=self.dilations[l],
                               shortcut=False, act=self.act)
            self.filters *= 2
        if self.max_pool_last:
            x = self.max_pool2(x)
        return x
