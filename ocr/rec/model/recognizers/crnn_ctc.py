# -*- coding: utf-8 -*-
# @Time : 2021/8/6 上午9:53
# @Author: yl
# @File: crnn_ctc.py
import tensorflow.keras as keras
from ..backbones import build_backbone
from ocr.rec.model.losses import build_loss
from ..encoders import build_encoder
from ..decoders import build_decoder


class CRNNNet(keras.Model):
    def __init__(self,
                 backbone=dict(type='ResNet'),
                 encoder=None,
                 decoder=None,
                 pretrained=None):
        super(CRNNNet, self).__init__()

        self.backbone = build_backbone(backbone)
        self.encoder = build_encoder(encoder) if encoder else None

        self.decoder = build_decoder(decoder)
        if pretrained:
            self.load_weights(pretrained, by_name=True, skip_mismatch=True)

    def call(self, x, training=None, mask=None):
        x = self.backbone(x, training)
        if self.encoder:
            x = self.encoder(x, training)
        output = self.decoder(x, training)
        return output

    def forward_train(self, data, y_true):
        y_pred = self(data)
        # y_true, label_length = label['label'], label['label_length']
        loss = self.loss(y_true, y_pred)
        return y_pred, loss
