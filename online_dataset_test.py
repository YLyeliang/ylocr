# -*- coding: utf-8 -*-
# @Time : 2021/9/8 下午3:22
# @Author: yl
# @File: online_dataset_test.py
import multiprocessing
import os
import pickle
import time

import numpy as np

from ocr.rec.data.online_dataset import OnlineDataSetV2


def generate_mp(dataset, f, num, iter_num):
    f = open(f, 'w')
    iter = 0
    s = time.time()
    for train, label in dataset.next_single():
        img_name = f"{str(num).rjust(5, '0')}.jpg"
        train.save(os.path.join(out_root, img_name))
        f.write(f"{img_name}\t{label}")
        num += 1
        iter += 1
        if iter == iter_num - 1:
            break
        if iter % 1000 == 0:
            e = time.time()
            cost = e - s
            s = time.time()
            print(f"process: {iter}. Time:{cost}")
    f.close()


if __name__ == '__main__':
    # dict_path = "utils/ppocr_keys_v1.txt"
    dict_path = "utils/pp_tianchi_char.txt"
    dicts = []
    with open(dict_path, 'r') as f:
        for p in f.readlines():
            p = p.replace("\n", "")
            dicts.append(p)
        if " " not in dicts:
            dicts.append(" ")
    char_idx_dict = {p: i for i, p in enumerate(dicts)}
    idx_char_dict = {i: p for i, p in enumerate(dicts)}

    strings_path = "data/wiki_corpus.pkl"
    with open(strings_path, 'rb') as f:
        strings = pickle.load(f)

    font_root = "trdg/fonts/"
    font_type = ['cn', 'latin', 'cn_tra']
    fonts = {}
    for type in font_type:
        ttfs = os.listdir(os.path.join(font_root, type))
        fonts_list = [os.path.join(font_root, type, ttf) for ttf in ttfs]
        fonts[type] = fonts_list
    bg_image_dir = ['data/crop_debug', 'data/crop_receipt']
    bg_image_weight = [0.2, 0.8]
    dataset = OnlineDataSetV2(char_idx_dict, strings, max_sequence_len=32, fonts=fonts, bg_image_dir=bg_image_dir,
                              bg_image_weight=bg_image_weight)

    # dataset = DatasetBuilder(char_idx_dict, generator=dataset)
    # data_loader = dataset(8, True)
    s = time.time()
    out_root = 'data/generation_train2'
    out_txt = "data/generation_train2.txt"
    num_process = 2
    total_num = 1e3
    num_per_process = int(total_num / num_process)
    processes = []
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    for i in range(num_process):
        txt = out_root + f"_{i + 1}.txt"
        start_num = int(num_per_process * i)
        process = multiprocessing.Process(target=generate_mp, args=(dataset, txt, 0, num_per_process))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    os.system("cat %s* > %s" % (out_root + "_", out_txt))

    # single
    # num = 1
    # f = open("data/generation_train.txt", 'w')
    # for train, label in dataset.next_single():
    #     img_name = f"{str(num).rjust(5, '0')}.jpg"
    #     train.save(os.path.join(out_root, img_name))
    #     f.write(f"{img_name}\t{label}")
    #     e = time.time()
    #     print(e - s)
    #     s = time.time()
    #     num += 1
    # plt.figure()
    # plt.imshow(train)
    # plt.show()
    # debug = 1
    # s = time.time()
    # for train, label in data_loader:
    #     # e = time.time()
    #     # print(e - s)
    #     # s = time.time()
    #     a = (np.array(train[0]) * 255).astype(np.uint8)
    #     char = tf.sparse.to_dense(label, -1).numpy()[0]
    #     string = [idx_char_dict[c] for c in char if c != -1]
    #     print(string)
    #     plt.figure()
    #     plt.imshow(a)
    #     plt.show()
    #     debug = 1
