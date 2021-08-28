# -*- coding: utf-8 -*-
# @Time : 2021/8/20 下午4:55
# @Author: yl
# @File: offline_train_fit.py

import pickle

from ocr.rec.model.recognizers import build_recognizer
from ocr.rec.model.losses import build_loss
from ocr.rec.core.evaluation import build_metric
from ocr.rec.data.offline_datasetV2 import OfflineDataset
from ocr.rec.model.converters import build_converter
from ocr.rec.data.simple_dataset import SimpleDataset
import tensorflow as tf
import tensorflow.keras as keras
import os
import matplotlib.pyplot as plt
import numpy as np

recognizer_config = dict(
    model=dict(type='CRNNNet',
               img_shape=(32, 320, 3),
               backbone=dict(
                   type='ResNetV2',
                   depth=34,
                   strides=((1, 1), (2, 1), (2, 1), (2, 1))
               ),
               encoder=None,
               decoder=dict(
                   type='SequenceDecoder',
                   decoder_type='rnn',
                   num_classes=None,
               ),
               ),
    loss=dict(
        type='CTCLayer',
        name="ctc_loss"
    ),
    converter=dict(
        type='CTCLabelConverter',
        char_idx_dict=None,
        character_type='ch',
    ),
    metric=[
        dict(type='LineAcc'), dict(type='NormEditDistance')],
    train_cfg=dict(
        batch_per_card=128,
    )
)


@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        logits = model(inputs["input"], training=True)
        loss_value = loss(inputs, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def ctc_func(args):
    y_true, y_pred, label_length = args
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Lambda, Dense, Bidirectional

CuDNNLSTM = tf.compat.v1.keras.layers.CuDNNLSTM
from tensorflow.keras import backend as K


def simple_model(inputs):
    conv_1 = Conv2D(32, (3, 3), activation="selu", padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)  # 16 64

    conv_2 = Conv2D(64, (3, 3), activation="selu", padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)  # 8 32

    conv_3 = Conv2D(128, (3, 3), activation="selu", padding='same')(pool_2)
    conv_4 = Conv2D(128, (3, 3), activation="selu", padding='same')(conv_3)

    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)  # 4 32

    conv_5 = Conv2D(256, (3, 3), activation="selu", padding='same')(pool_4)

    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(256, (3, 3), activation="selu", padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)  # 2 32

    conv_7 = Conv2D(64, (2, 2), activation="selu")(pool_6)  # 1 31

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)  # 31,512

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(squeezed)
    blstm_2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(blstm_1)

    softmax_output = Dense(len(char_idx_dict.keys()) + 1, activation='softmax', name="dense")(blstm_2)
    return softmax_output


if __name__ == '__main__':

    import string

    chars = string.ascii_letters
    chars += "0123456789 "
    # chars += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~ "
    char_idx_dict = {p: i for i, p in enumerate(chars)}
    idx_char_dict = {i: p for i, p in enumerate(chars)}

    # dict_path = "utils/ppocr_keys_v1.txt"
    # dicts = []
    # with open(dict_path, 'rb') as f:
    #     for p in f.readlines():
    #         p = p.decode('utf-8').strip("\n").strip("\r\n")
    #         dicts.append(p)
    #     dicts.append(" ")
    # char_idx_dict = {p: i for i, p in enumerate(dicts)}
    # idx_char_dict = {i: p for i, p in enumerate(dicts)}

    model_cfg = recognizer_config['model']
    model_cfg['decoder']['num_classes'] = len(char_idx_dict.keys()) + 1

    recognizer_config['converter']['char_idx_dict'] = char_idx_dict
    batch_size = recognizer_config['train_cfg']['batch_per_card']
    data_loader = SimpleDataset(char_idx_dict, img_shape=(32, 320, 3),
                                img_paths=['../spark-ai-summit-2020-text-extraction/mjsynth_sample', ],
                                lab_paths=['../spark-ai-summit-2020-text-extraction/mjsynth.txt'])

    data_loader = data_loader(batch_size, True, True, num_workers=4)

    # model = build_recognizer(model_cfg)

    loss = build_loss(recognizer_config['loss'])

    model_prefix = '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'

    x = keras.layers.Input(shape=(32, 320, 3), batch_size=None, name='input')
    # label = keras.layers.Input(name="label", shape=(None,), dtype="float32")
    # label_length = keras.layers.Input(name="label_length", shape=(1), dtype="int64")
    # input_length = keras.layers.Input(name='input_length', shape=(1), dtype="int64")

    output = simple_model(x)
    # ctc_loss = loss(label, output, label_length)
    # ctc_loss = keras.layers.Lambda(ctc_func, output_shape=(1,), name='ctc')([label, output, label_length])

    model = keras.Model(inputs=x, outputs=output)

    # model = build_model(len(char_idx_dict)+1)
    # model = keras.Model(inputs=x, outputs=output)
    recognizer_config['converter']['char_idx_dict'] = char_idx_dict
    converter = build_converter(recognizer_config['converter'])

    metrics = [build_metric(m) for m in recognizer_config['metric']]

    lr = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=5000)

    # optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True, clipvalue=5.)
    optimizer = keras.optimizers.Adam(lr, clipnorm=1.0)

    model.compile(optimizer=optimizer)

    model.summary()

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("my_model/result.hdf6", save_weights_only=True),
    # ]
    # model.fit(data_loader,
    #           # steps_per_epoch=1000,
    #           epochs=100,
    #           callbacks=callbacks,
    #           )
    # model.load_weights("my_model/save.hdf5")
    for epoch in range(30):
        for iter, inputs in enumerate(data_loader):
            # train = inputs["input"]
            # label = inputs["label"]
            # a = (np.array(train[0]) * 255).astype(np.uint8)
            # d = list(label.numpy())[0]
            # string = [idx_char_dict[p] for p in d if p in idx_char_dict]
            # print(string)
            # plt.figure()
            # plt.imshow(a)
            # plt.show()

            # with tf.GradientTape() as tape:
            #     logits = model(inputs["input"], training=True)
            #     loss_value = loss(inputs, logits)
            # grads = tape.gradient(loss_value, model.trainable_weights)
            # optimizer.apply_gradients(zip(grads, model.trainable_weights))
            loss_value = train_step(inputs)
            # lr = keras.backend.get_value(optimizer.learning_rate)
            # for metric in metrics:
            #     metric.update_state(inputs["label"], logits)
            #     print(metric.result())
            if (iter + 1) % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}. Iter: {iter + 1}. Loss: {loss_value.numpy()}.")
                # f"Line acc: {metrics[0].result().numpy():.5f}. Norm edit dis: {metrics[1].result().numpy():.5f}")
            # if (iter +1) %100==0:
        model.save_weights("my_model/save.hdf5")
