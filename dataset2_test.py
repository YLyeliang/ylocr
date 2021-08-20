# -*- coding: utf-8 -*-
# @Time : 2021/8/17 下午4:30
# @Author: yl
# @File: dataset2_test.py
import os
import pickle
import time

import numpy as np

from ocr.rec.data.online_dataset import OnlineDataSetV2, DatasetBuilder
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
    idx_char_dict = {i: p for i, p in enumerate(dicts)}

    strings_path = "data/wiki_corpus.pkl"
    with open(strings_path, 'rb') as f:
        strings = pickle.load(f)

    font_root = "trdg/fonts/"
    font_type = ['cn', 'latin']
    fonts = {}
    for type in font_type:
        ttfs = os.listdir(os.path.join(font_root, type))
        fonts_list = [os.path.join(font_root, type, ttf) for ttf in ttfs]
        fonts[type] = fonts_list
    dataset = OnlineDataSetV2(char_idx_dict, strings, max_sequence_len=32, fonts=fonts, bg_image_dir='data/crop_debug')

    dataset = DatasetBuilder(char_idx_dict, generator=dataset)
    data_loader = dataset(8, True)
    # for train ,label,label_length in dataset():
    #     debug = 1
    s = time.time()
    for train, label in data_loader:
        # e = time.time()
        # print(e - s)
        # s = time.time()
        a = (np.array(train[0]) * 255).astype(np.uint8)
        char = tf.sparse.to_dense(label, -1).numpy()[0]
        string = [idx_char_dict[c] for c in char if c != -1]
        print(string)
        plt.figure()
        plt.imshow(a)
        plt.show()
        debug = 1
