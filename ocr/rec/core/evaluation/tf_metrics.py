# -*- coding: utf-8 -*-
# @Time : 2021/8/16 上午9:49
# @Author: yl
# @File: tf_metrics.py


import tensorflow as tf
import tensorflow.keras as keras


class LineAcc(keras.metrics.Metric):
    def __init__(self, name='line_acc', **kwargs):
        super(LineAcc, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        def sparse2dense(tensor, shape):
            tensor = tf.sparse.reset_shape(tensor, shape)
            tensor = tf.sparse.to_dense(tensor, default_value=-1)
            tensor = tf.cast(tensor, tf.float32)
            return tensor

        y_true_shape = tf.shape(y_true)
        batch_size = y_true_shape[0]
        y_pred_shape = tf.shape(y_pred)
        max_width = tf.math.maximum(y_true_shape[1], y_pred_shape[1])
        logit_length = tf.fill([batch_size], y_pred_shape[1])

        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)

        y_true = sparse2dense(y_true, [batch_size, max_width])
        y_pred = sparse2dense(decoded[0], [batch_size, max_width])

        num_errors = tf.math.reduce_any(
            tf.math.not_equal(y_true, y_pred), axis=1)
        num_errors = tf.cast(num_errors, tf.float32)
        num_errors = tf.math.reduce_sum(num_errors)
        batch_size = tf.cast(batch_size, tf.float32)
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)

    def result(self):
        return self.count / self.total

    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)


class NormEditDistance(keras.metrics.Metric):
    def __init__(self, name='norm_edit_distance', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sum_distance = self.add_weight(name='sum_distance',
                                            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        logit_length = tf.fill([batch_size], y_pred_shape[1])

        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)

        sum_distance = tf.math.reduce_sum(tf.edit_distance(tf.cast(decoded[0],tf.int32), y_true, normalize=True))
        batch_size = tf.cast(batch_size, tf.float32)

        self.sum_distance.assign_add(sum_distance)
        self.total.assign_add(batch_size)

    def result(self):
        return 1 - self.sum_distance / self.total

    def reset_states(self):
        self.sum_distance.assign(0)
        self.total.assign(0)


if __name__ == '__main__':
    norm = NormEditDistance()
    y_true = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2], [2, 1]], values=[1, 2, 2], dense_shape=[3, 4])
    y_pred = tf.sparse.SparseTensor(indices=[[0, 1], [1, 3], [2, 1]], values=[1, 2, 3], dense_shape=[3, 4])
    y_pred_shape = tf.shape(y_pred)
    batch_size = y_pred_shape[0]
    logit_length = tf.fill([batch_size], y_pred_shape[1])

    # decoded, _ = tf.nn.ctc_greedy_decoder(
    #     inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
    #     sequence_length=logit_length)

    # sum_distance = tf.math.reduce_sum(tf.edit_distance(y_pred, y_true))
    # norm.update_state(y_true, y_pred)
    debug = 1
