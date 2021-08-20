# -*- coding: utf-8 -*-
# @Time : 2021/8/20 下午4:55
# @Author: yl
# @File: offline_train_fit.py

import pickle

from ocr.rec.model.recognizers import build_recognizer
from ocr.rec.model.losses import build_loss
from ocr.rec.core.evaluation import build_metric
from ocr.rec.data.offline_dataset import  OfflineDataSet,DatasetBuilder
import tensorflow as tf
import tensorflow.keras as keras
import os

recognizer_config = dict(
    model=dict(type='CRNNNet',
               backbone=dict(
                   type='ResNetV2',
               ),
               encoder=None,
               decoder=dict(
                   type='SequenceDecoder',
                   decoder_type='rnn',
                   num_classes=None,
               ),
               ),
    loss=dict(
        type='CTCLoss',
        logits_time_major=False,
        blank_index=-1
    ),
    converter=dict(
        type='CTCLabelConverter',
        char_idx_dict=None,
        character_type='cn',
        use_space_char=True
    ),
    metric=[
        dict(type='LineAcc'), dict(type='NormEditDistance')],
    train_cfg=dict(
        batch_per_card=4,
    )
)

if __name__ == '__main__':

    dict_path = "utils/ppocr_keys_v1.txt"
    dicts = []
    with open(dict_path, 'rb') as f:
        for p in f.readlines():
            p = p.decode('utf-8').strip("\n").strip("\r\n")
            dicts.append(p)
        dicts.append(" ")
    char_idx_dict = {p: i for i, p in enumerate(dicts)}
    idx_char_dict = {i: p for i, p in enumerate(dicts)}
    model_cfg = recognizer_config['model']
    model_cfg['decoder']['num_classes'] = len(char_idx_dict.keys()) + 1

    recognizer_config['converter']['char_idx_dict'] = char_idx_dict

    dataset = OfflineDataSet(char_idx_dict, max_sequence_len=32, img_path=['../spark-ai-summit-2020-text-extraction/mjsynth_sample', ], lab_path=['../spark-ai-summit-2020-text-extraction/mjsynth.txt'])
    data_loader = DatasetBuilder(char_idx_dict, generator=dataset, img_shape=(32, 320, 3))

    data_loader = data_loader(16, True)

    batch_size = recognizer_config['train_cfg']['batch_per_card']
    # strategy = tf.distribute.MirroredStrategy()
    # batch_size = recognizer_config['train_cfg']['batch_size_per_card'] * strategy.num_replicas_in_sync

    model = build_recognizer(model_cfg)
    # from ocr.rec.model.recognizers.crnn_ctc import CRNN

    # model.build(input_shape=(8, 32, 320, 3))

    model_prefix = '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'

    x = tf.keras.layers.Input(shape=(32, 320, 3), batch_size=None)
    output = model(x)
    model = keras.Model(inputs=x, outputs=output)
    # model = build_model(len(char_idx_dict)+1)
    # model = keras.Model(inputs=x, outputs=output)
    # converter = build_converter(recognizer_config['converter'])

    loss = build_loss(recognizer_config['loss'])

    metrics = [build_metric(m) for m in recognizer_config['metric']]

    lr = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=1000)

    # optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True, clipvalue=5.)
    optimizer = keras.optimizers.Adam(lr)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint("my_model/result", save_weights_only=True),
    ]
    model.fit(data_loader,
              steps_per_epoch=1000,
              epochs=100,
              callbacks=callbacks,
              )
