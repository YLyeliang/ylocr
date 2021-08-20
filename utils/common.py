# -*- coding: utf-8 -*-
# @Time : 2021/8/17 下午5:10
# @Author: yl
# @File: common.py

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
