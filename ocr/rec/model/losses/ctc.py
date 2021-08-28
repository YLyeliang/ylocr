# -*- coding: utf-8 -*-
# @Time : 2021/8/9 下午5:03
# @Author: yl
# @File: ctc.py
import tensorflow.keras as keras

import tensorflow as tf


# class CTCLayer:
#     def __init__(self):
#         self.loss_fn = keras.backend.ctc_batch_cost
#
#     def __call__(self, y_pred, y_true, label_length):
#         # Compute the training-time losses value and add it
#         # to the layer using `self.add_loss()`.
#
#         batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#         input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#
#         input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#
#         loss = self.loss_fn(y_true, y_pred, input_length, label_length)
#         # self.add_loss(loss)
#
#         # At test time, just return the computed predictions
#         return loss

class CTCLayer(keras.losses.Loss):

    def __init__(self, name='ctc_loss'):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, inputs, y_pred):
        # Compute the training-time losses value and add it
        # to the layer using `self.add_loss()`.
        y_true = inputs["label"]
        label_length = inputs["label_length"]
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        # label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        # label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        # self.add_loss(loss)

        # At test time, just return the computed predictions
        return loss


class CTCLoss(keras.losses.Loss):
    def __init__(self, logits_time_major=False, blank_index=-1, name='ctc_loss'):
        """

        Args:
            logits_time_major: If False (default) , shape is [batch, time, logits],
                If True, logits is shaped [time, batch, logits].
            blank_index: Set the class index to use for the blank label. default is
                -1 (num_classes - 1).
        """
        super(CTCLoss, self).__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        """
        Calculate CTC loss.
        Args:
            y_true:
            y_pred:

        Returns:

        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred_shape = tf.shape(y_pred)
        logit_length = tf.fill([y_pred_shape[0]], y_pred_shape[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.math.reduce_mean(loss)
