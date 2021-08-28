# -*- coding: utf-8 -*-
# @Time : 2021/8/10 下午4:26
# @Author: yl
# @File: dataset_test.py
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

from ocr.rec.data.offline_datasetV2 import DatasetBuilder, OfflineDataset
from ocr.rec.data.simple_dataset import SimpleDataset
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

    # dataset = DatasetBuilder(char_idx_dict, max_sequence_len=32, img_paths=['data', ],
    #                          lab_paths=['data/train_list.txt', ])
    # dataset = OfflineDataset(char_idx_dict, max_sequence_len=40, img_paths=['data', ],
    #                          lab_paths=['data/train_list.txt', ])
    # data_loader = dataset(4, True)

    data_loader = SimpleDataset(char_idx_dict, img_shape=(32, 320, 3),
                                img_paths=['../spark-ai-summit-2020-text-extraction/mjsynth_sample', ],
                                lab_paths=['../spark-ai-summit-2020-text-extraction/mjsynth.txt'])

    data_loader = data_loader(256, True, True, num_workers=6)

    s = time.time()
    for iter, inputs in enumerate(data_loader):
        e = time.time()
        cost = e - s
        train = inputs["input"]
        label = inputs["label"]
        # a = (np.array(train[0]) * 255).astype(np.uint8)
        d = list(label.numpy())[0]
        string = [idx_char_dict[p] for p in d if p in idx_char_dict]
        print(f"{iter}: batch_size:{len(label.numpy())} string[0]: {string} time: {cost}")
        # plt.figure()
        # plt.imshow(a)
        # plt.show()
        debug = 1
        s = time.time()
