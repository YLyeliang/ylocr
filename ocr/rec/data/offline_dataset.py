# -*- coding: utf-8 -*-
# @Time : 2021/8/12 上午10:51
# @Author: yl
# @File: offline_dataset.py


import tensorflow.keras as keras
import numpy as np
import cv2
import os
import codecs


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

    def load_annotations(self, img_path, lab_path, char_idx_dict):
        img_names = []
        label_names = []
        if isinstance(img_path, str) and isinstance(lab_path, str):

    def load_samples(self, image_root, label_root, char_idx_dict):
        label_list = os.listdir(label_root)
        sample_list = []
        for label_name in label_list:
            label_path = os.path.join(label_root, label_name)
            img_path = os.path.join(image_root, label_name[:-4] + ".jpg")
            try:
                with codecs.open(label_path, "rb", encoding='utf-8') as label_file:
                    txt = label_file.readline()
                    txt = txt.strip()
                    flag = False
                    for char in txt:
                        if char not in char_idx_dict:
                            flag = True
                            break
                    if flag:
                        continue
                    sample_list.append([img_path, txt])
            except:
                print("Error test sample:", label_name)
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
