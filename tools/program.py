# -*- coding: utf-8 -*-
# @Time : 2021/8/26 下午3:05
# @Author: yl
# @File: program.py


import yaml
import os
import pickle
import string


def loadCharDict(file_path, character_type='ch', use_space_char=True):
    support_character_type = [
        'ch', 'en', 'EN_symbol'
    ]

    assert character_type in support_character_type
    dicts = []
    if file_path:
        _, ext = os.path.splitext(file_path)
        assert ext in ['.txt', '.pkl']
        if ext.endswith('.txt'):
            with open(file_path, 'rb') as f:
                for p in f.readlines():
                    p = p.decode('utf-8').strip("\n").strip("\r\n")
                    dicts.append(p)
            char_idx_dict = {p: i for i, p in enumerate(dicts)}
            idx_char_dict = {i: p for i, p in enumerate(dicts)}
        elif ext.endswith('.pkl'):
            with open(file_path, 'rb')as f:
                idx_char_dict, char_idx_dict = pickle.load(f)
    else:
        if character_type == 'en':
            chars = string.ascii_letters
            chars += "0123456789 "
        elif character_type == 'EN_Symbol':
            chars = string.ascii_letters
            chars += "0123456789 "
            chars += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~ "
        else:
            raise NotImplementedError
        char_idx_dict = {p: i for i, p in enumerate(chars)}
        idx_char_dict = {i: p for i, p in enumerate(chars)}

    if use_space_char and " " not in char_idx_dict.keys():
        char_idx_dict[" "] = len(char_idx_dict.keys())
        idx_char_dict[len(idx_char_dict.keys())] = " "

    return char_idx_dict, idx_char_dict


def mergeConfig(cfg, dicts):
    global_cfg = cfg["Global"]
    new_global_cfg = global_cfg.copy()
    for key, val in dicts.items():
        if val:
            new_global_cfg.update({key: val})
    cfg['Global'] = new_global_cfg
    return cfg


def loadConfig(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    cfg = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return cfg


if __name__ == '__main__':
    config = "../configs/rec/crnn/crnn_res50.yaml"
