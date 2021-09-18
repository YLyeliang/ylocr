# -*- coding: utf-8 -*-
# @Time : 2021/8/17 上午11:04
# @Author: yl
# @File: normal_test.py
# import tensorflow as tf
import cv2

if __name__ == '__main__':
    # dict_path = "utils/ppocr_keys_v1.txt"
    # dicts = []
    # with open(dict_path, 'rb') as f:
    #     for p in f.readlines():
    #         p = p.decode('utf-8').strip("\n").strip("\r\n")
    #         dicts.append(p)
    #     dicts.append(" ")
    # char_idx_dict = {p: i for i, p in enumerate(dicts)}
    #
    # # a = tf.random.normal(shape=[10,10,3])
    # # b= tf.image.resize(a,(4,4))
    # # c= cv2.resize(a.numpy(),(4,4),interpolation=cv2.INTER_LINEAR)
    # table = tf.lookup.StaticHashTable(
    #     tf.lookup.KeyValueTensorInitializer(tf.convert_to_tensor(list(char_idx_dict.keys())),
    #                                         tf.convert_to_tensor(list(char_idx_dict.values()))),default_value=-1)
    # # lamb = lambda
    # rt = tf.ragged.constant([['哈', 'b', 'c'], [], ['d', 'e'], ['g']])
    # chars = tf.strings.unicode_split(rt,"UTF-8")
    #
    # rt_c = tf.ragged.map_flat_values(table.lookup, rt)
    #
    # debug = 1

    point_text= {'pointsList': [[1006.0, 802.0, 1153.0, 802.0, 1153.0, 846.0, 1006.0, 846.0],
                    [286.0, 686.0, 506.0, 686.0, 506.0, 754.0, 286.0, 754.0],
                    [285.0, 563.0, 383.0, 563.0, 383.0, 635.0, 285.0, 635.0],
                    [289.0, 448.0, 380.0, 448.0, 380.0, 513.0, 289.0, 513.0],
                    [283.0, 326.0, 414.0, 322.0, 416.0, 393.0, 285.0, 396.0]],
     'transcriptionsList': ['奇泡', 'PREVIEW', 'YOU', 'EOR', 'SONG']}

    points = point_text['pointsList']
    texts = point_text['transcriptionsList']
    for i, point in enumerate(zip(points, texts)):
        debug =1
        p = point[0]
        p = ",".join(list(map(str,p)))
        # print(f"{i + 1}\t{','.join(point)}\t{text}\n")