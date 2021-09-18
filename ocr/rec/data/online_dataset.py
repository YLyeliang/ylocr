# -*- coding: utf-8 -*-
# @Time : 2021/8/10 下午3:14
# @Author: yl
# @File: online_dataset.py

import tensorflow.keras as keras
import numpy as np
import cv2
from trdg.apis.text_gen import textImgGen
import tensorflow as tf

try:
    AUTOTUNE = tf.data.AUTOTUNE
except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class OnlineDataSet:
    def __init__(self,
                 char_idx_dict,
                 strings,
                 max_sequence_len,
                 fonts,
                 bg_image_dir,
                 img_size=(32, 320, 3),
                 batch_size=64,
                 blank_index=-1,
                 ):
        """

        Args:
            char_idx_dict:
            batch_size:
            img_h:
            strings:
            max_sequence_len:
            fonts:
            bg_image_dir:
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.char_idx_dict = char_idx_dict
        self.chars_list = list(self.char_idx_dict.keys())
        self.strings = strings
        if blank_index == -1:
            self.blank_label = self.get_output_size() - 1
        else:
            self.blank_label = blank_index
        self.max_sequence_len = max_sequence_len
        self.generator = textImgGen(img_h=img_size[0], char_idx_dict=char_idx_dict,
                                    strings=strings, absolute_max_string_len=max_sequence_len, fonts=fonts,
                                    bg_image_dir=bg_image_dir)

    def get_output_size(self):  # 输出尺寸，所有字符+空白
        return len(self.char_idx_dict) + 1

    def get_batch(self, batch_size):
        imgs = []
        img_w_list = []
        labels = np.ones([batch_size, self.max_sequence_len]) * self.blank_label
        label_length = np.zeros([batch_size, 1])
        text_list = []
        for i in range(batch_size):
            img, text = self.generator.next()

            img = np.asarray(img)
            real_w = img.shape[1]

            if self.img_size[1] and real_w > self.img_size[1]:  # 固定尺寸条件下
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
                real_w = self.img_size[1]
            img = img.astype(np.float)
            img = img / 255.

            imgs.append(img)
            text_list.append(text)
            img_w_list.append(real_w)

            # 汉字转数字
            label = [self.char_idx_dict[char] for char in text]
            label_len = len(label)
            labels[i, :label_len] = label

            label_length[i] = label_len

        if self.img_size[1]:  # 如果宽不为None，则为固定尺寸，否则每个批次的训练样本随当前批次图像的最大宽度设置
            img_w = self.img_size[1]
        else:
            img_w = np.max(img_w_list)
        if keras.backend.image_data_format() == 'channels_first':
            data = np.zeros([batch_size, 3, self.img_size[1], img_w])
        else:
            data = np.zeros([batch_size, self.img_size[0], img_w, 3])

        for idx in range(len(img_w_list)):
            if keras.backend.image_data_format() == 'channels_first':
                data[idx, :, :, img_w_list[idx]] = imgs[idx]
            else:
                data[idx, :, :img_w_list[idx], :] = imgs[idx]

        return data, labels, label_length

    def next_train(self):
        while True:
            ret = self.get_batch(self.batch_size)
            yield ret


class OnlineDataSetV2:
    def __init__(self,
                 char_idx_dict,
                 strings,
                 max_sequence_len,
                 fonts,
                 bg_image_dir,
                 img_size=(32, 320, 3),
                 bg_image_weight=[1, 1],
                 blank_index=-1,
                 ):
        """

        Args:
            char_idx_dict:
            img_h:
            strings:
            max_sequence_len:
            fonts:
            bg_image_dir:
        """
        self.img_size = img_size
        self.char_idx_dict = char_idx_dict
        self.chars_list = list(self.char_idx_dict.keys())
        self.strings = strings
        self.max_sequence_len = max_sequence_len
        self.generator = textImgGen(img_h=img_size[0], char_idx_dict=char_idx_dict,
                                    strings=strings, absolute_max_string_len=max_sequence_len, fonts=fonts,
                                    bg_image_dir=bg_image_dir, bg_image_weight=bg_image_weight)

    def next_single(self):
        while True:
            yield self.generator.next()  # img, text

    def get_batch(self, batch_size):
        imgs = []
        img_w_list = []
        labels = np.ones([batch_size, self.max_sequence_len]) * self.blank_label
        label_length = np.zeros([batch_size, 1])
        text_list = []
        for i in range(batch_size):
            img, text = self.generator.next()

            img = np.asarray(img)
            real_w = img.shape[1]

            if self.img_size[1] and real_w > self.img_size[1]:  # 固定尺寸条件下
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
                real_w = self.img_size[1]
            img = img.astype(np.float)
            img = img / 255.

            imgs.append(img)
            text_list.append(text)
            img_w_list.append(real_w)

            # 汉字转数字
            label = [self.char_idx_dict[char] for char in text]
            label_len = len(label)
            labels[i, :label_len] = label

            label_length[i] = label_len

        if self.img_size[1]:  # 如果宽不为None，则为固定尺寸，否则每个批次的训练样本随当前批次图像的最大宽度设置
            img_w = self.img_size[1]
        else:
            img_w = np.max(img_w_list)
        if keras.backend.image_data_format() == 'channels_first':
            data = np.zeros([batch_size, 3, self.img_size[1], img_w])
        else:
            data = np.zeros([batch_size, self.img_size[0], img_w, 3])

        for idx in range(len(img_w_list)):
            if keras.backend.image_data_format() == 'channels_first':
                data[idx, :, :, img_w_list[idx]] = imgs[idx]
            else:
                data[idx, :, :img_w_list[idx], :] = imgs[idx]

        return data, labels, label_length

    def next_train(self):
        while True:
            ret = self.get_batch(self.batch_size)
            yield ret


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
                                              output_shapes=((self.img_shape[0], None, self.img_shape[2]), ()))

    def preprocess(self, img, label):
        if self.aspect_ratio:
            img_shape = tf.shape(img)  # h w c
            scale_factor = self.img_shape[0] / img_shape[0]
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
            img_width = tf.cast(img_width, tf.int32)
        else:
            img_width = self.img_shape[1]
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
