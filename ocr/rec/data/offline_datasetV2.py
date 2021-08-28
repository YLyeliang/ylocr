# -*- coding: utf-8 -*-
# @Time : 2021/8/23 下午1:46
# @Author: yl
# @File: offline_datasetV2.py

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
                 img_paths=[],
                 lab_paths=[],
                 add_space=True,
                 max_sequence_len=32):
        self.img_shape = img_shape
        self.char_idx_dict = char_idx_dict
        self.aspect_ratio = aspect_ratio
        if blank_index == -1:
            self.blank_label = self.num_classes - 1
        else:
            self.blank_label = blank_index

        self.img_paths = img_paths
        self.lab_paths = lab_paths
        self.img_list, self.lab_list = self.load_samples(img_paths, lab_paths, char_idx_dict)

        self.max_sequence_len = max_sequence_len
        self.char_idx_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.convert_to_tensor(list(char_idx_dict.keys())),
                                                tf.convert_to_tensor(list(char_idx_dict.values()))), default_value=-1)

    @property
    def num_classes(self):
        return len(self.char_idx_dict) + 1

    def load_samples(self, image_root, label_list, char_idx_dict):
        images = []
        labels = []
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
                if flag or len(label) == 0:
                    continue
                img_path = os.path.join(image_root[i], img_name)
                if not os.path.isfile(img_path):
                    continue
                # sample_list.append([img_path, label])
                images.append(img_path)
                labels.append(label)
        return images, labels

    def preprocess(self, img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)

        # 2. Decode
        img = tf.io.decode_jpeg(img, channels=3)

        if self.aspect_ratio:
            img_shape = tf.shape(img)  # h w c
            scale_factor = self.img_shape[0] / img_shape[0]  # t_h / h
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)  # t_h / h * c_w
            img_width = tf.cast(img_width, tf.int32)  # final w
        else:
            img_width = self.img_shape[1]
        img_width = tf.minimum(tf.cast(self.img_shape[1], tf.int32), img_width) if self.img_shape[2] else img_width
        img = tf.image.resize(img, (self.img_shape[0], img_width)) / 255.0
        return img, label

    def _filter_img(self, img, label):
        img_shape = tf.shape(img)
        return img_shape[1] <= self.img_shape[1]

    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.char_idx_table.lookup, chars)
        tokens = tokens.to_sparse()
        return imgs, tokens

    def __call__(self, batch_size, is_training):
        ds = tf.data.Dataset.from_tensor_slices((self.img_list, self.lab_list))

        if is_training:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self.preprocess, AUTOTUNE)  # pre-process
        if self.aspect_ratio and batch_size != 1:
            # ds = ds.filter(self._filter_img)  # filter
            ds = ds.padded_batch(batch_size, padded_shapes=((32, self.img_shape[1], 3), ()),
                                 drop_remainder=True)  # padded
        else:
            ds = ds.batch(batch_size)
        ds = ds.map(self._tokenize, AUTOTUNE)
        ds.prefetch(AUTOTUNE)
        return ds


class OfflineDataset(object):
    def __init__(self,
                 char_idx_dict,
                 img_shape=(32, 320, 3),
                 aspect_ratio=True,
                 blank_index=-1,
                 img_paths=[],
                 lab_paths=[],
                 use_space_char=True,
                 max_sequence_len=32):
        self.img_shape = img_shape

        self.aspect_ratio = aspect_ratio
        self.max_sequence_len = max_sequence_len
        if blank_index == -1:
            self.blank_label = self.num_classes - 1
        else:
            self.blank_label = blank_index

        if use_space_char and " " not in char_idx_dict.keys():
            char_idx_dict[" "] = len(char_idx_dict.keys())

        self.char_idx_dict = char_idx_dict
        self.img_paths = img_paths
        self.lab_paths = lab_paths
        self.img_list, self.lab_list, self.lab_len_list = self.load_samples(img_paths, lab_paths, char_idx_dict)

        self.char_idx_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.convert_to_tensor(list(char_idx_dict.keys())),
                                                tf.convert_to_tensor(list(char_idx_dict.values()))), default_value=-1)

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
            for line in lines:
                img_name, label = line.rstrip('\n').split('\t')
                flag = False
                encoded_label = np.full((self.max_sequence_len), fill_value=self.num_classes, dtype=np.int32)
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
        img = tf.io.decode_jpeg(img, channels=3)

        if self.aspect_ratio:
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

    def _tokenize(self, imgs, labels, label_lengths):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.map_fn(self.char_idx_table.lookup, chars)
        # tokens = tf.ragged.map_flat_values(self.char_idx_table.lookup, chars)
        # tokens = tokens.to_sparse()
        return imgs, tokens, label_lengths

    def __call__(self, batch_size, is_training):
        # ds = tf.data.Dataset.from_tensor_slices(
        #     {"input": self.img_list, "label": self.lab_list, "label_length": self.lab_len_list})
        ds = tf.data.Dataset.from_tensor_slices(
            (self.img_list, self.lab_list, self.lab_len_list))
        # debug = list(ds.as_numpy_iterator())
        if is_training:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self.preprocess, AUTOTUNE)  # pre-process
        if self.aspect_ratio and batch_size != 1:
            # ds = ds.filter(self._filter_img)  # filter
            ds = ds.padded_batch(batch_size, padded_shapes=((32, self.img_shape[1], 3), (None,), (None,)),
                                 drop_remainder=True)  # padded
        else:
            ds = ds.batch(batch_size)
        ds = ds.map(lambda x, y, z: {"input": x, "label": y, "label_length": z}, AUTOTUNE)
        # ds = ds.map(self._tokenize, AUTOTUNE)
        ds.prefetch(AUTOTUNE)
        return ds
