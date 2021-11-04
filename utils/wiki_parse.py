# -*- coding: utf-8 -*-
# @Time : 2021/8/14 上午11:23
# @Author: yl
# @File: wiki_parse.py
"""
解码wiki语料库
"""
import os
import pickle
import time

from tqdm import tqdm

out_file = "../data/wiki_corpus.pkl"

in_folder = "/home/yel/dataset/wiki_corpus/text/"


def parse_wiki():
    fw = open(out_file, "wb")
    all_Lines = []
    for root, dirs, files in os.walk(in_folder):
        for file in tqdm(files):
            if not os.path.isfile(os.path.join(root, file)):
                continue

            fr = open(os.path.join(root, file), 'r')
            lines = fr.readlines()
            lines = filter(lambda x: '</doc>' not in x and '<doc' not in x and len(x) > 0,
                           [line.strip() for line in lines])
            lines = list(lines)
            # lines = [line + '\n' for line in lines]
            all_Lines += lines
            fr.close()
    pickle.dump(all_Lines, fw)
    fw.close()


def filter_pkl(src, out):
    """
    对语料u
    Returns:

    """
    fw = open(out, 'wb')
    all_lines = []
    with open(src, 'rb') as f:
        data = pickle.load(f)
    for line in tqdm(data):
        if len(line) < 5 or len(line) > 30:
            continue
        else:
            all_lines.append(line)
    pickle.dump(all_lines, fw)
    fw.close()


def read_pkl(out_file):
    with open(out_file, 'rb') as f:
        dat = pickle.load(f)

    char = "鹿晗"
    index = dat.index(char)
    lists = dat[index:index+1000]
    flag = char in dat
    debug = 1
    return dat


if __name__ == '__main__':
    # parse_wiki()
    data = read_pkl('../data/tianchi_corpus.pkl')
    # filter_pkl(out_file, '../data/wiki_corpus_filter.pkl')
