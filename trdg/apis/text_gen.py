# -*- coding: utf-8 -*-
# @Time : 2021/8/4 上午9:45
# @Author: yl
# @File: text_gen.py
import os
import random

from ..data_generator import FakeTextDataGenerator
import numpy as np
from ..string_generator import create_strings_randomly
from ..utils import text_content_gen, full_to_half
from fontTools.ttLib import TTFont


class textImgGen:
    """
    合成数据生成的主流程：
    1-> 随机生成文本： 1）中文：语料库随机抽取; 2) 随机生成：字母、数字和标点
    2-> 判断生成文本是否均在字体中,消除不在的;消除后判断长度，为空时重新生成
    3-> 随机生成文本生成时的其他参数：歪斜，模糊，扭曲，垂直/水平，字符间距，空格间距，文本与边缘距离/是否完全贴边，描边宽度，
    Args:
        img_h:
        char_idx_dict:
        strings:
        absolute_max_string_len:
        fonts:
        size:
        bg_image_dir:
        **generator_kwargs:
    """

    def __init__(
            self,
            char_idx_dict,
            strings,
            absolute_max_string_len,
            extra_strings=None,
            fonts=[],
            img_h=32,
            bg_image_dir=None,
            bg_image_weight=[1, 1],
            seed=None,
    ):

        self.img_h = img_h
        self.char_idx_dict = char_idx_dict
        self.chars = list(self.char_idx_dict.keys())
        self.strings = strings
        self.extra_strings = extra_strings
        self.extra_strings_num = len(extra_strings) if extra_strings else 0
        self.strings_num = len(strings)
        self.fonts = fonts
        self.fonts_num = len(fonts)

        self.blank_label = len(self.char_idx_dict) - 1
        self.absolute_max_string_len = absolute_max_string_len
        self.generated_count = 0
        self.bg_image_dir = bg_image_dir
        self.bg_image_weight = bg_image_weight
        assert len(bg_image_dir) == len(bg_image_weight)
        self.cache_bg_image_list()
        self.fonts_dict = {}
        if seed:
            np.random.seed(seed)
            random.seed(seed)

    def cache_bg_image_list(self):
        bg_image_lists = []
        for image_dir in self.bg_image_dir:
            files = os.listdir(image_dir)
            files = [os.path.join(image_dir, file) for file in files]
            bg_image_lists.append(files)
        self.bg_image_lists = bg_image_lists

    def string_generate(self, fonts):
        """
        生成文本内容
        """
        random_num = random.random()
        if random_num > 0.4:
            fonts = fonts['cn']
        elif random_num > 0.3:
            fonts = fonts['cn_tra']
        else:
            fonts = fonts['cn'] + fonts['latin']

        font = fonts[random.randint(0, len(fonts) - 1)]

        if random_num > 0.4:  # 简体
            random_num2 = random.random()
            if random_num2 > 0.6:
                string = self.strings[random.randint(0, self.strings_num - 1)]
                while len(string) < 10:
                    string += self.strings[random.randint(0, self.strings_num - 1)]
            elif random_num2 > 0.2 and self.extra_strings:
                string = self.extra_strings[random.randint(0, self.extra_strings_num - 1)]
                while len(string) < 10:
                    string += self.extra_strings[random.randint(0, self.extra_strings_num - 1)]
            else:
                string = "".join(random.sample(self.chars, 15))

            text = text_content_gen(string,
                                    self.char_idx_dict, flag='sim', count=(8, 15))
        elif random_num > 0.3:  # 繁体
            string = self.strings[random.randint(0, self.strings_num - 1)]
            while len(string) < 10:
                string += self.strings[random.randint(0, self.strings_num - 1)]
            text = text_content_gen(string,
                                    self.char_idx_dict, flag='tra', count=(8, 15))
        elif random_num > 0.2:  # 纯数字
            text = create_strings_randomly(2, allow_variable=True, count=1, let=False, num=True, sym=False, lang='en')
            text = "".join(text)
            self.gen_kwargs.update(dict(distorsion_type=0, orientation=0))
        elif random_num > 0.1:  # 纯字母
            text = create_strings_randomly(2, allow_variable=True, count=1, let=True, num=False, sym=False,
                                           lang='en')
            text = "".join(text)
            self.gen_kwargs.update(dict(distorsion_type=0, orientation=0))
        else:  # 数字+字母+符号
            text = create_strings_randomly(2, allow_variable=True, count=1, let=True, num=True, sym=True,
                                           lang='en')
            text = "".join(text)
            self.gen_kwargs.update(dict(distorsion_type=0, orientation=0))
        text = full_to_half(text)  # 全角转半角
        # 判断字符是否都在字体里
        text = self.valid_char(text, font)
        text = text.strip()  # 去除字符收尾两端的空格
        return text, font

    def valid_char(self, string, font):
        """
        消除不在字体里的字符
        """
        if font not in self.fonts_dict:
            ttfont = TTFont(font)
            uniMap = ttfont['cmap'].tables[0].ttFont.getBestCmap()
            self.fonts_dict[font] = uniMap
        return ''.join([char for char in string if ord(char) in self.fonts_dict[font]])

    def aug_param_generate(self):
        """
        随机生成文本增强参数
        """
        self.gen_kwargs = dict(
            skewing_angle=random.randint(0, 1),
            random_skew=random.randint(0, 1),
            blur=0,
            random_blur=True,
            background_type=np.random.choice([1, 3, 4], p=[0.15, 0.15, 0.7]),
            distorsion_type=0,  # 0: No 1: sin 2: cos
            distorsion_orientation=random.randint(0, 1),  # 0: vertical 1: horizontal 2: both
            is_handwritten=False,
            name_format=0,
            width=-1,
            alignment=np.random.randint(3),  # 0: left 1: center 2: right, latter two should set width first
            text_color="#282828,#FFFFFF",
            orientation=np.random.choice([0, 1], p=[0.8, 0.2]),  # 0: horizontal 1: vertical
            space_width=round(np.random.rand() + 1, 1),  # random 1-2  the width of " " x factor
            character_spacing=random.randint(0, 2),  # the width between characters
            margins=np.random.randint(1, 4, 4).tolist(),
            output_mask=False,
            fit=random.randint(0, 1),  # 0: not tight the text 1: crop the text region
            word_split=False,
            stroke_width=random.randint(0, 1),  # 笔画宽度
            stroke_fill="#282828,#FFFFFF",
            max_width=640)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self.generated_count += 1
        # 文本内容生成
        # 选择字体
        self.aug_param_generate()
        text, font = self.string_generate(self.fonts)
        while len(text) < 5:
            text, font = self.string_generate(self.fonts)

        # 有些字体在draw时会报错， allocation array size too large
        text_img = None
        while not text_img:
            try:
                text_img, text = FakeTextDataGenerator.generate(
                    self.generated_count,
                    text,
                    font,
                    None,
                    self.img_h,
                    None,
                    image_dir=np.random.choice(self.bg_image_lists, p=self.bg_image_weight),
                    **self.gen_kwargs
                )
            except:
                pass
            if text_img:
                return (text_img, text)
            else:
                text, font = self.string_generate(self.fonts)
