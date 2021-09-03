# -*- coding: utf-8 -*-
# @Time : 2021/8/26 下午2:01
# @Author: yl
# @File: simple_dataset.py

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

try:
    AUTOTUNE = tf.data.AUTOTUNE
except Exception:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class SimpleDataset(object):
    def __init__(self,
                 char_idx_dict,
                 img_shape=(32, 320, 3),
                 keep_aspect_ratio=True,
                 blank_index=-1,
                 img_paths=[],
                 lab_paths=[],
                 filter_shape=None,
                 max_sequence_len=32):
        self.char_idx_dict = char_idx_dict

        img_shape = [None if 'None' == num else num for num in img_shape]
        self.img_shape = img_shape

        self.keep_aspect_ratio = keep_aspect_ratio
        self.max_sequence_len = max_sequence_len
        if blank_index == -1:
            self.blank_label = self.num_classes - 1
        else:
            self.blank_label = blank_index

        self.filter_shape = filter_shape

        self.img_paths = img_paths
        self.lab_paths = lab_paths
        self.img_list, self.lab_list, self.lab_len_list = self.load_samples(img_paths, lab_paths, char_idx_dict)
        # print(len(self.lab_list))

        # self.char_idx_table = tf.lookup.StaticHashTable(
        #     tf.lookup.KeyValueTensorInitializer(tf.convert_to_tensor(list(char_idx_dict.keys())),
        #                                         tf.convert_to_tensor(list(char_idx_dict.values()))), default_value=-1)

    @property
    def num_classes(self):
        return len(self.char_idx_dict) + 1

    def min_ctc_len(self, text):
        ctc_len = 1
        for i in range(1, len(text)):
            if text[i] == text[i - 1]:
                ctc_len += 2
            else:
                ctc_len += 1
        return ctc_len

    def load_samples(self, image_root, label_list, char_idx_dict):
        images = []
        labels = []
        label_lengths = []
        for i, label_txt in enumerate(label_list):
            with open(label_txt, 'r') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                img_name, label = line.rstrip('\n').split('\t')
                flag = False
                encoded_label = np.full((self.max_sequence_len), fill_value=self.blank_label, dtype=np.int32)
                if len(label) > self.max_sequence_len:
                    continue
                for j, char in enumerate(label):
                    if char not in char_idx_dict:
                        flag = True
                        break
                    encoded_label[j] = self.char_idx_dict[char] * 1.0

                if flag or len(label) == 0 or self.min_ctc_len(label) + 1 > self.max_sequence_len - 1:
                    continue
                img_path = os.path.join(image_root[i], img_name)
                if not os.path.isfile(img_path):
                    continue

                label_length = [len(label), ]
                images.append(img_path)
                labels.append(encoded_label)
                label_lengths.append(label_length)
        return images, labels, label_lengths

    def preprocess(self, img_path, labels, label_lengths):

        # 1. Read image
        img = tf.io.read_file(img_path)

        # 2. Decode
        img = tf.io.decode_jpeg(img, channels=self.img_shape[2])

        if self.keep_aspect_ratio:
            img_shape = tf.shape(img)  # h w c
            scale_factor = self.img_shape[0] / img_shape[0]  # t_h / h
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)  # t_h / h * c_w
            img_width = tf.cast(img_width, tf.int32)  # final w
        else:
            img_width = self.img_shape[1]
        if self.img_shape[1]:
            img_width = tf.minimum(tf.cast(self.img_shape[1], tf.int32), img_width) if self.img_shape[2] else img_width
        img = tf.image.resize(img, (self.img_shape[0], img_width)) / 255.0
        return img, labels, label_lengths

    def _filter_img(self, img, label, label_lengths):
        img_shape = tf.shape(img)
        return img_shape[1] <= self.img_shape[1]

    def __call__(self, batch_per_card, shuffle, drop_remainder, num_workers):

        num_workers = num_workers if num_workers else AUTOTUNE
        ds = tf.data.Dataset.from_tensor_slices(
            (self.img_list, self.lab_list, self.lab_len_list))
        ds = tf.data.Dataset.range(1).interleave(
            lambda _: ds,
            num_parallel_calls=num_workers
        )
        # debug = list(ds.as_numpy_iterator())
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self.preprocess, num_workers)  # pre-process
        if self.keep_aspect_ratio and batch_per_card != 1:
            # ds = ds.filter(self._filter_img)  # filter
            ds = ds.padded_batch(batch_per_card,
                                 padded_shapes=((32, self.img_shape[1], self.img_shape[2]), (None,), (None,)),
                                 drop_remainder=drop_remainder)  # padded
        else:
            ds = ds.batch(batch_per_card)
        ds = ds.map(lambda x, y, z: {"input": x, "label": y, "label_length": z}, num_workers)
        # ds = ds.map(self._tokenize, AUTOTUNE)
        ds.prefetch(AUTOTUNE)
        return ds, len(self.img_list) // batch_per_card
