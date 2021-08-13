# -*- coding: utf-8 -*-
# @Time : 2021/8/10 下午4:26
# @Author: yl
# @File: dataset_test.py
import os

import numpy as np

from ocr.rec.data.online_dataset import OnlineDataSet
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dict_path = "utils/ppocr_keys_v1.txt"
    dicts = []
    with open(dict_path, 'r') as f:
        for p in f.readlines():
            p = p.replace("\n", "")
            dicts.append(p)
        dicts.append(" ")
    char_idx_dict = {p: i for i, p in enumerate(dicts)}
    strings = ["哈哈", 'Hhasf', "sdjflksdjlkjl", 'haha sfjslal']
    font_root = "trdg/fonts/"
    font_type = ['cn', 'latin']
    fonts = {}
    for type in font_type:
        ttfs = os.listdir(os.path.join(font_root, type))
        fonts_list = [os.path.join(font_root, type, ttf) for ttf in ttfs]
        fonts[type] = fonts_list
    dataset = OnlineDataSet(char_idx_dict, strings, max_sequence_len=32, fonts=fonts, bg_image_dir='data/crop_debug',
                            batch_size=8).next_train
    data_loader = tf.data.Dataset.from_generator(
        dataset,
        output_signature=(
            tf.TensorSpec(shape=(None, 32, 320, 3), name='train_data'),
            tf.TensorSpec(shape=(None, 32), dtype=tf.int64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64)
        )).map(lambda x, y, z: (x, {'label': y, "length": z}), num_parallel_calls=4).prefetch(buffer_size=8)
    # for train ,label,label_length in dataset():
    #     debug = 1
    for train, label in data_loader:
        a = (np.array(train[0]) * 255).astype(np.uint8)
        plt.figure()
        plt.imshow(a)
        plt.show()
        debug = 1
