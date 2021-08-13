# -*- coding: utf-8 -*-
# @Time : 2021/8/9 下午5:24
# @Author: yl
# @File: ctc.py

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import string


class BaseLabelConverter(object):
    """

    Args:
        char_idx_dict:
        character_type:
        use_space_char:

    Attributes:
         character_type(str): cn en EN_symbol
         dict(dict): the char_idx dict
         character(list): the list of characters

    """

    def __init__(self,
                 char_idx_dict,
                 character_type='cn',
                 use_space_char=False,
                 ignored_index=-1):

        support_character_type = [
            'cn', 'en', 'EN_symbol'
        ]
        assert character_type in support_character_type, f"Not supported character type: {character_type}"

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = ""
            for key, val in char_idx_dict.items():
                self.character_str += key
            if use_space_char and " " not in char_idx_dict:
                self.character_str += " "
            dict_character = list(self.character_str)
        else:
            raise NotImplementedError

        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)  # 是否添加空字符
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        if ignored_index == -1:
            self.ignored_index = len(self.dict) # 空白字符
        else:
            self.ignored_index = ignored_index

    def add_special_char(self, dict_character):
        return dict_character

    def index2str(self, text_index, text_prob=None, is_remove_duplicate=False):
        """
        convert text-index into text-label.
        Args:
            text_index(np.ndarray): with shape (N,T)
            text_prob(np.ndarray |None): with shape (N,T)
            is_remove_duplicate(bool): Whether remove the duplicate character, True for test

        Returns:
            result_list(list): [text, conf] where conf is the mean of all text probabilities
        """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [self.ignored_index]  # for ctc blank


class CTCLabelConverter(BaseLabelConverter):
    def __init__(self,
                 char_idx_dict=None,
                 character_type='cn',
                 use_space_char=False):
        super(CTCLabelConverter, self).__init__(char_idx_dict, character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        """

        Args:
            preds:
            label:
            *args:
            **kwargs:

        Returns:

        """
        if isinstance(preds, tf.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)
        text = self.index2str(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.index2str(label)
        return text, label

    # Not useful for now
    # def add_special_char(self, dict_character):
    #     dict_character = dict_character+['blank']
    #     return dict_character
