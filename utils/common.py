# -*- coding: utf-8 -*-
# @Time : 2021/8/17 下午5:10
# @Author: yl
# @File: common.py
import os
import numpy as np


def full_to_half(string):
    half_str = ""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        uchar = chr(inside_code)
        half_str += uchar
    return half_str


def listFromTxt(txt, start_line=0, end_col=3, filter=None):
    """
    读取比赛数据的txt,其中格式如下:
    id \t bbox \t text
    Args:
        txt:
        start_line:
        end_col:
        filter:
    """
    with open(txt, 'r') as f:
        lines = f.readlines()[start_line:]
        labels = [[] for _ in range(end_col)]
        for line in lines:
            line = line.rstrip('\n').split('\t')[:end_col]
            bbox = line[1].split(',')
            bbox = np.array([float(_) for _ in bbox]).reshape(4, 2)
            line[1] = bbox
            if end_col > 2:
                if line[2] == "*":
                    continue
            for i, item in enumerate(line):
                labels[i].append(item)
    return labels
