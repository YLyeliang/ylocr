# -*- coding: utf-8 -*-
# @Time : 2021/9/15 上午10:41
# @Author: yl
# @File: data_gen_mp.py

import multiprocessing
import os
import pickle
import time

from trdg.apis.text_gen import textImgGen


def generate_mp(dataset, f, num, iter_num):
    f = open(f, 'w')
    iter = 0
    s = time.time()
    for train, label in dataset:
        img_name = f"{str(num).rjust(7, '0')}.jpg"
        train.save(os.path.join(out_root, img_name))
        f.write(f"{img_name}\t{label}\n")
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
    with open(dict_path, 'rb') as f:
        for p in f.readlines():
            p = p.decode('utf-8').strip('\n').strip('\r\n')
            dicts.append(p)
        if " " not in dicts:
            dicts.append(" ")
    char_idx_dict = {p: i for i, p in enumerate(dicts)}
    # idx_char_dict = {i: p for i, p in enumerate(dicts)}

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
    data_generator = textImgGen(img_h=32, char_idx_dict=char_idx_dict,
                                strings=strings, absolute_max_string_len=32, fonts=fonts,
                                bg_image_dir=bg_image_dir, bg_image_weight=bg_image_weight)

    # dataset = DatasetBuilder(char_idx_dict, generator=dataset)
    # data_loader = dataset(8, True)
    s = time.time()
    out_root = 'data/generation_train4'
    out_txt = "data/generation_train4.txt"
    num_process = 1
    total_num = 1e2
    num_per_process = int(total_num / num_process)
    processes = []
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    for i in range(num_process):
        txt = out_txt.split('.')[0] + f"_{i + 1}.txt"
        start_num = int(num_per_process * i)
        process = multiprocessing.Process(target=generate_mp, args=(data_generator, txt, start_num, num_per_process))
        process.start()
        print("start: ", i)
        processes.append(process)

    for process in processes:
        process.join()

    os.system("cat %s* > %s" % (out_txt.split('.')[0] + "_", out_txt))
    os.system(f"rm -f {out_txt.split('.')[0]}_*")
