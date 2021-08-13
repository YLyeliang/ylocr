# -*- coding: utf-8 -*-
# @Time : 2021/8/10 下午3:14
# @Author: yl
# @File: online_dataset.py

import tensorflow.keras as keras
import numpy as np
import cv2
from trdg.apis.text_gen import textImgGen


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
        self.generator = textImgGen(batch_size=batch_size, img_h=img_size[0], char_idx_dict=char_idx_dict,
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
