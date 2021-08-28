# -*- coding: utf-8 -*-
# @Time : 2021/8/6 上午9:53
# @Author: yl
# @File: crnn_ctc.py
import tensorflow.keras as keras
from ..backbones import build_backbone
from ocr.rec.model.losses import build_loss
from ..encoders import build_encoder
from ..decoders import build_decoder
from ..core.conv_utils import ConvBlock, BottleNeck
from tensorflow.keras import layers


class CRNNNet(object):
    def __init__(self,
                 img_shape,
                 backbone=dict(type='ResNet'),
                 encoder=None,
                 decoder=None,
                 pretrained=None):
        super(CRNNNet, self).__init__()

        img_shape = [None if num == 'None' else num for num in img_shape]
        self.img_shape = img_shape

        self.backbone = build_backbone(backbone)
        self.encoder = build_encoder(encoder) if encoder else None

        self.decoder = build_decoder(decoder)
        self.pretrained = pretrained
        # if pretrained:
        #     self.load_weights(pretrained, by_name=True, skip_mismatch=True)

    def __call__(self):
        inputs = keras.layers.Input(shape=self.img_shape, batch_size=None)
        x = self.backbone(inputs)
        if self.encoder:
            x = self.encoder(x)
        output = self.decoder(x)
        model = keras.Model(inputs=inputs, outputs=output)
        return model

    # def forward_train(self, data, y_true):
    #     y_pred = self(data)
    # y_true, label_length = label['label'], label['label_length']
    # loss = self.loss(y_true, y_pred)
    # return y_pred, loss

# def stem(input_tensor):
#     x = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu',
#                       kernel_initializer='he_normal')(input_tensor)
#     x = ConvBlock(x, 3, 64, strides=(2, 2), kernel_initializer='he_normal', act='relu', name='1')
#     return x
#
#
# def CRNN(input_tensor, num_classes=6625, filters=64, stage_blocks=[3, 4, 6, 3],
#          strides=((1, 1), (2, 2), (2, 1), (2, 1)), dilations=(1, 1, 1, 1), act='relu'):
#     x = stem(input_tensor)
#     for l, strides in enumerate(strides):
#         x = BottleNeck(x, 3, filters, stage=l + 2, block='a', dilation=dilations[l], strides=strides,
#                        shortcut=True)
#         for i in range(stage_blocks[l] - 1):
#             x = BottleNeck(x, 3, filters, stage=l + 2, block=chr(ord('b') + i), dilation=dilations[l],
#                            shortcut=False, act=act)
#         filters *= 2
#
#     x = layers.Permute((2, 1, 3), name='permute')(x)
#     x = layers.TimeDistributed(layers.Flatten(), name="flatten")(x)  # flatten h c into one channel
#     x = layers.Dropout(rate=0.2)(x)
#     y_pred = layers.TimeDistributed(layers.Dense(num_classes, input_shape=(None,)), name='fc')(x)
#     return y_pred
