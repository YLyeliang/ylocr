# -*- coding: utf-8 -*-
# @Time : 2021/8/7 上午11:33
# @Author: yl
# @File: statistic.py
"""
数据统计：字符统计
"""

import cv2
import pickle
from tqdm import tqdm
import pandas as pd
import json
from tensorflow.keras.optimizers import Optimizer, SGD


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


def aspect_ratio(img_path_list):
    ratios = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        h, w, c = img.shape
        ratio = w / h
        ratios.append(ratio)

    sorted(ratios)
    return ratios


def get_chars():
    train = [
        pd.read_csv("../data/OCR复赛数据集01.csv"),
        pd.read_csv("../data/OCR复赛数据集02.csv")
    ]
    df = pd.concat(train)
    for row in df.iterrows():
        labels = json.loads(row[1]['融合答案'])[0]
        for label in labels:
            text = json.loads(label['text'])['text']
            for char in text:
                if char not in global_dict.keys():
                    global_dict[char] = 1
                else:
                    global_dict[char] += 1


def writeChars(char_list, out_txt, out_pkl):
    if out_txt:
        with open(out_txt, 'w') as f:
            f.writelines(char_list)

    if out_pkl:
        with open(out_pkl, 'wb') as f:
            idx_char_dict = {i: p.rstrip() for i, p in enumerate(char_list)}
            char_idx_dict = {p.rstrip(): i for i, p in enumerate(char_list)}
            pickle.dump((char_idx_dict, idx_char_dict), f)


def mergeTxt(src_txts, dst_txt):
    with open(src_txts[0], 'r') as f:
        lines = f.readlines()
    for src_txt in src_txts[1:]:
        with open(src_txt, 'r') as f:
            new_lines = f.readlines()
        for line in new_lines:
            if line not in lines:
                lines.append(line)
    with open(dst_txt, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    # CharFrequencyFromPkl('../data/wiki_corpus.pkl', "../data/wiki_char_frequency.txt")
    # with open("../data/wiki_char_frequency.txt", 'r') as f:
    #     lines = f.readlines()

    # 统计训练集中出现的字符
    # global_dict = {}
    # csv_file1 = "../data/OCR复赛数据集01.csv"
    # csv_file2 = "../data/OCR复赛数据集02.csv"
    # get_chars()
    # global_dict = sorted(global_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
    # char_list = []
    # for char in global_dict:
    #     char_list.append(char[0] + '\n')
    #
    # writeChars(char_list, "tianchi_char.txt", "tianchi_char.pkl")

    # 合并字符集
    src_txts = ['ppocr_keys_v1.txt', 'tianchi_char.txt']
    out_txts = "pp_tianchi_char.txt"
    mergeTxt(src_txts, out_txts)
