"""
Utility functions
"""

import os

import numpy as np
from hanziconv import HanziConv
from fontTools.ttLib import TTFont
import pickle


def load_dict(lang):
    """Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(
            os.path.join(os.path.dirname(__file__), "dicts", lang + ".txt"),
            "r",
            encoding="utf8",
            errors="ignore",
    ) as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
    return lang_dict


def load_fonts(lang):
    """Load all fonts in the fonts directories
    """

    if lang in os.listdir(os.path.join(os.path.dirname(__file__), "fonts")):
        return [
            os.path.join(os.path.dirname(__file__), "fonts/{}".format(lang), font)
            for font in os.listdir(
                os.path.join(os.path.dirname(__file__), "fonts/{}".format(lang))
            )
        ]
    else:
        return [
            os.path.join(os.path.dirname(__file__), "fonts/latin", font)
            for font in os.listdir(
                os.path.join(os.path.dirname(__file__), "fonts/latin")
            )
        ]


def read_sentence(pkl_path):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            sentence_list = pickle.load(file)
    return sentence_list


def valid_sentence(string, char_idx_dict):
    """
    消除不在词库里的字符
    """
    return ''.join([char for char in string if char in char_idx_dict])


def valid_char(string, font):
    """
    消除不在字体里的字符
    """
    font = TTFont(font)
    uniMap = font['cmap'].tables[0].ttFont.getBestCmap()
    return ''.join([char for char in string if ord(char) in uniMap])


def text_content_gen(text, char_idx_dict, flag, count=(1, 30)):
    if flag == 'tra':
        text = HanziConv.toTraditional(text)  # Traditional
    else:
        text = text

    valid_text = valid_sentence(text, char_idx_dict)  # 字符都在词库里
    text_len = np.random.randint(count[0], count[1])
    if len(valid_text) <= text_len:
        return valid_text
    start = np.random.randint(len(valid_text) - text_len)
    return valid_text[start:start + text_len]


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
