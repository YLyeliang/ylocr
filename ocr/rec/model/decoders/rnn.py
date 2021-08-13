# -*- coding: utf-8 -*-
# @Time : 2021/8/9 下午3:37
# @Author: yl
# @File: rnn.py
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

CuDNNLSTM = tf.compat.v1.keras.layers.CuDNNLSTM


class Im2Seq(keras.layers.Layer):
    def __init__(self):
        super(Im2Seq, self).__init__()

    def call(self, x):
        B, H, W, C = x.shape
        assert H == 1
        x = tf.squeeze(x, axis=1)  # N W C
        return x


class DecoderWithRNN(keras.Sequential):
    def __init__(self, units=128):
        super(DecoderWithRNN, self).__init__()
        self.add(layers.Bidirectional(CuDNNLSTM(units, return_sequences=True)), name="lstm_1")
        self.add(layers.Bidirectional(CuDNNLSTM(units, return_sequences=True)), name="lstm_2")


class DecoderWithFC(keras.Sequential):
    def __init__(self, units):
        super(DecoderWithFC, self).__init__()
        self.add(layers.TimeDistributed(layers.Dense(units)), name='fc')


class SequenceDecoder(keras.Model):
    """
    Sequence decode part, 3 choices:
    reshape -> fc / rnn
    1. reshape: reshape NHWC into NWC, where H is 1
    2. fc: perform fc for features, where feature shape is NWC
    3. rnn: perform bidirectional LSTM to features, where feature shape is NWC
    Args:
        decoder_type:
        hidden_size:
        num_classes:
    """

    def __init__(self, decoder_type, hidden_size=256, num_classes=None):

        super(SequenceDecoder, self).__init__()
        self.num_classes = num_classes
        self.decoder_reshape = Im2Seq()
        if decoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_decoder_dict = {
                'reshape': Im2Seq,
                'fc': DecoderWithFC,
                'rnn': DecoderWithRNN
            }
            assert decoder_type in support_decoder_dict, '{} must in {}'.format(
                decoder_type, support_decoder_dict.keys())
            self.decoder = support_decoder_dict[decoder_type](256)
            self.only_reshape = False
        self.pred = layers.Dense(num_classes, activation='softmax', name='dense')

    def call(self, x, training=None, mask=None):
        x = self.decoder_reshape(x)
        if not self.only_reshape:
            x = self.decoder(x)
        pred = self.pred(x)
        return pred


class StepDecoder(keras.Model):
    """
    the feature shape from backbone is N 2 W C
    Args:
        num_classes(int): the classes numbers
    """

    def __init__(self,
                 num_classes=None):
        super(StepDecoder, self).__init__()
        self.num_classes = num_classes
        self.permute = layers.Permute((2, 1, 3), name='permute')
        self.flatten = layers.TimeDistributed(layers.Flatten(), name="flatten")  # flatten h c into one channel

        self.drop_out = layers.Dropout(rate=0.2)
        self.dense = layers.TimeDistributed(layers.Dense(num_classes), name='fc')
        self.softmax = tf.nn.softmax

    def call(self, x, training=None, mask=None):
        x = self.permute(x)
        x = self.flatten(x)
        if training:
            x = self.drop_out(x)

        x = self.dense(x)
        pred = self.softmax(x, axis=-1)
        return pred
