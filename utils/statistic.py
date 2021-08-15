# -*- coding: utf-8 -*-
# @Time : 2021/8/7 上午11:33
# @Author: yl
# @File: statistic.py
"""
数据统计：字符统计
"""

import pickle
from tqdm import tqdm


def CharFrequencyFromPkl(pkl_file, out_file=None):
    char_fre = {}
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    for line in tqdm(data):
        for char in line:
            if char_fre.get(char, None):
                char_fre[char] += 1
            else:
                char_fre[char] = 1

    if out_file:
        with open(out_file, 'w') as f:
            for key, val in char_fre.items():
                f.write(f"{key}: {val}\n")
        print(len(char_fre.keys()))


if __name__ == '__main__':
    # CharFrequencyFromPkl('../data/wiki_corpus.pkl', "../data/wiki_char_frequency.txt")
    with open("../data/wiki_char_frequency.txt", 'r') as f:
        lines = f.readlines()

    #     debug = 1
