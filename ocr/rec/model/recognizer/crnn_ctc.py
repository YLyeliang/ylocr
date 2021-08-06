# -*- coding: utf-8 -*-
# @Time : 2021/8/6 上午9:53
# @Author: yl
# @File: crnn_ctc.py
import tensorflow as tf
import tensorflow.keras as keras
from ..backbone.resnet import ResNet


class CRNNNet(object):
    def __init__(self,
                 backbone=dict(type='ResNet'),
                 encoder=None,
                 decoder=None,
                 loss=None,
                 label_converter=None,
                 max_seq_len=32,
                 pretrained=None):
        super(CRNNNet, self).__init__()

        self.backbone = eval(backbone.pop("type"))(**backbone)

