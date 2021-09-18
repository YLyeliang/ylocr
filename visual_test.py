# -*- coding: utf-8 -*-
# @Time : 2021/9/6 下午6:14
# @Author: yl
# @File: visual_test.py
# img = []
import base64

import numpy as np
import cv2


# img = np.array(img).reshape(-1, 150).astype(np.uint8)
# img = cv2.resize(img, (202, 300))
# cv2.imwrite("data/recovery/7.jpg", img)
# cv2.imshow("img", img)
# cv2.waitKey()


def fromTxt(txt):
    txt = "log.txt"
    with open(txt, 'r')  as f:
        lines = f.readlines()
    arr = []
    count = 1
    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        if len(line) <= 66:  # 小于等于 127.0.0.1 ....
            continue
        if len(arr) == 0 and not line.startswith('['):  # 中间截断只剩后面一半
            continue

        str_split = line.split("127.0.0.1")  # 去掉127的部分
        if len(str_split) == 1:
            img = str_split[0]
            arr.append(img)
        else:  # 出现127
            img = "".join([str_split[0], str_split[-1][57:]])
            arr.append(img)
        # 如果是前面一半
        if '[' in img and ']' in img:  # 完整数组
            img = eval(img)
            arr = []
        elif ']' not in img:  # 前面一半
            continue
        else:  # 后面一半
            img = "".join(arr)
            img = eval(img)
            arr = []
        shape = eval(lines[i + 1])[:2] if i + 1 < len(lines) else (150, 150)
        img = np.array(img).reshape(-1, 150).astype(np.uint8)
        img = cv2.resize(img, shape[::-1])
        cv2.imwrite(f"data/recovery/{count}.jpg", img)
        count += 1
        # cv2.imshow("img", img)
        # cv2.waitKey()

def fromTxtBase64(txt):
    txt = "data/log_18.txt"
    with open(txt, 'r')  as f:
        lines = f.readlines()
    arr = []
    count = 607
    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        if "please use" in line:
            continue
        if len(line) <= 66:  # 小于等于 127.0.0.1 ....
            continue
        if len(arr) == 0 and not line.startswith('['):  # 中间截断只剩后面一半
            continue

        str_split = line.split("127.0.0.1")  # 去掉127的部分
        if len(str_split) == 1:
            base64_list = str_split[0]
            arr.append(base64_list)
        else:  # 出现127
            base64_list = "".join([str_split[0], str_split[-1][57:]])
            arr.append(base64_list)
        # 如果是前面一半
        if '[' in base64_list and ']' in base64_list:  # 完整数组
            base64_list = eval(base64_list)
            arr = []
        elif ']' not in base64_list:  # 前面一半
            continue
        else:  # 后面一半
            base64_list = "".join(arr)
            base64_list = eval(base64_list)
            arr = []

        for base64_string in base64_list:
            img = base64.b64decode(base64_string)
            # img = np.array(img_string).tostring()
            img = np.frombuffer(img, dtype=np.uint8)
            img = cv2.imdecode(img,1)
            # cv2.imshow("img", img)
            # cv2.waitKey()
            cv2.imwrite(f"data/recovery4/{count}.jpg", img)
            count += 1


if __name__ == '__main__':
    # fromTxt(0)
    fromTxtBase64(0)
