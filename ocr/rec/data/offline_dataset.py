# -*- coding: utf-8 -*-
# @Time : 2021/8/12 上午10:51
# @Author: yl
# @File: offline_dataset.py

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2
import os
import codecs

try:
    AUTOTUNE = tf.data.AUTOTUNE
except Exception:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset(tf.data.TextLineDataset):
    def __init__(self, filename, **kwargs):
        self.dirname = os.path.dirname(filename)
        super().__init__(filename, **kwargs)

    def parse_func(self, line):
        raise NotImplementedError

    def parse_line(self, line):
        line = tf.strings.strip(line)
        img_relative_path, label = self.parse_func(line)
        img_path = tf.strings.join([self.dirname, os.sep, img_relative_path])
        return img_path, label


class SimpleDataset(Dataset):

    def parse_func(self, line):
        splited_line = tf.strings.split(line)
        img_relative_path, label = splited_line[0], splited_line[1]
        return img_relative_path, label


class DatasetBuilder:
    def __init__(self,
                 char_idx_dict,
                 img_shape=(32, 320, 3),
                 aspect_ratio=True,
                 blank_index=-1,
                 max_sequence_len=32,
                 generator=None):
        self.img_shape = img_shape
        self.char_idx_dict = char_idx_dict
        self.aspect_ratio = aspect_ratio
        if blank_index == -1:
            self.blank_label = self.num_classes - 1
        else:
            self.blank_label = blank_index
        self.max_sequence_len = max_sequence_len
        self.generator = generator
        self.char_idx_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.convert_to_tensor(list(char_idx_dict.keys())),
                                                tf.convert_to_tensor(list(char_idx_dict.values()))), default_value=-1)

    @property
    def num_classes(self):
        return len(self.char_idx_dict) + 1

    def _filter_img(self, img, label):
        img_shape = tf.shape(img)
        return img_shape[1] <= self.img_shape[1]

    def parse_generator(self):
        return tf.data.Dataset.from_generator(self.generator.next_single, output_types=(tf.uint8, tf.string),
                                              output_shapes=((None, None, self.img_shape[2]), ()))

    def preprocess(self, img, label):
        if self.aspect_ratio:
            img_shape = tf.shape(img)  # h w c
            scale_factor = self.img_shape[0] / img_shape[0] # t_h / h
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64) # t_h / h * c_w
            img_width = tf.cast(img_width, tf.int32) # final w
        else:
            img_width = self.img_shape[1]
        # img_width = tf.maximum(self.img_shape[2], img_width) if self.img_shape[2] else img_width
        img = tf.image.resize(img, (self.img_shape[0], img_width)) / 255.0
        return img, label

    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.char_idx_table.lookup, chars)
        tokens = tokens.to_sparse()
        return imgs, tokens

    def __call__(self, batch_size, is_training):
        ds = self.parse_generator()
        ds = ds.map(self.preprocess, AUTOTUNE)
        if self.aspect_ratio and batch_size != 1:
            ds = ds.filter(self._filter_img)
            ds = ds.padded_batch(batch_size, padded_shapes=((32, self.img_shape[1], 3), ()))
        else:
            ds = ds.batch(batch_size)
        ds = ds.map(self._tokenize, AUTOTUNE)
        ds.prefetch(AUTOTUNE)
        return ds


class OfflineDataSet:
    def __init__(self,
                 char_idx_dict,
                 max_sequence_len,
                 img_size=(32, 320, 3),
                 batch_size=64,
                 blank_index=-1,
                 img_path=None,
                 lab_path=None
                 ):
        """

        Args:
            char_idx_dict:
            batch_size:
            max_sequence_len:
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.char_idx_dict = char_idx_dict
        self.chars_list = list(self.char_idx_dict.keys())
        if blank_index == -1:
            self.blank_label = self.get_output_size() - 1
        else:
            self.blank_label = blank_index
        self.max_sequence_len = max_sequence_len
        self.data_list = self.load_samples(img_path, lab_path, char_idx_dict)

    def load_samples(self, image_root, label_list, char_idx_dict):
        sample_list = []
        for i, label_txt in enumerate(label_list):
            with open(label_txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                img_name, label = line.rstrip('\n').split('\t')
                flag = False
                for char in label:
                    if char not in char_idx_dict:
                        flag = True
                        break
                if flag:
                    continue
                img_path = os.path.join(image_root[i], img_name)
                sample_list.append([img_path, label])
        return sample_list

    def min_ctc_len(self, text):
        ctc_len = 1
        for i in range(1, len(text)):
            if text[i] == text[i - 1]:
                ctc_len += 2
            else:
                ctc_len += 1
        return ctc_len

    def get_output_size(self):  # 输出尺寸，所有字符+空白
        return len(self.char_idx_dict) + 1

    def __len__(self):
        data_len = int(len(self.data_list) / self.batch_size)
        return data_len

    def next_single(self):
        for img_name, label in self.data_list:
            img = cv2.imread(img_name)
            yield img, label
