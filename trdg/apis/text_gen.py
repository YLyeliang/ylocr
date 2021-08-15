# -*- coding: utf-8 -*-
# @Time : 2021/8/4 上午9:45
# @Author: yl
# @File: text_gen.py
from ..data_generator import FakeTextDataGenerator
import numpy as np
from ..string_generator import create_strings_randomly
from ..utils import text_content_gen, valid_char


class textImgGen:
    """
    合成数据生成的主流程：
    1-> 随机生成文本： 1）中文：语料库随机抽取; 2) 随机生成：字母、数字和标点
    2-> 判断生成文本是否均在字体中,消除不在的;消除后判断长度，为空时重新生成
    3-> 随机生成文本生成时的其他参数：歪斜，模糊，扭曲，垂直/水平，字符间距，空格间距，文本与边缘距离/是否完全贴边，描边宽度，
    Args:
        batch_size:
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
            batch_size,
            char_idx_dict,
            strings,
            absolute_max_string_len,
            fonts=[],
            img_h=32,
            bg_image_dir=None,
    ):

        self.batch_size = batch_size
        self.img_h = img_h
        self.char_idx_dict = char_idx_dict
        self.chars = list(self.char_idx_dict.keys())
        self.strings = strings
        self.strings_num = len(strings)
        self.fonts = fonts
        self.fonts_num = len(fonts)

        self.blank_label = len(self.char_idx_dict) - 1
        self.absolute_max_string_len = absolute_max_string_len
        self.generated_count = 0
        self.bg_image_dir = bg_image_dir

    def string_generate(self, fonts):
        """
        生成文本内容
        """
        random_num = np.random.random()
        if random_num > 0.55:
            fonts = fonts['cn']
        # elif random_num > 0.45:
        #     fonts = fonts['font_tri']
        else:
            fonts = fonts['cn'] + fonts['latin']

        font = fonts[np.random.randint(len(fonts))]

        if random_num > 0.55:  # 简体
            text = text_content_gen(self.strings[np.random.randint(self.strings_num)],
                                    self.char_idx_dict, flag='sim', count=(2, 25))
        elif random_num > 0.45:  # 繁体
            text = text_content_gen(self.strings[np.random.randint(self.strings_num)],
                                    self.char_idx_dict, flag='tra', count=(2, 25))
        elif random_num > 0.25:  # 纯数字
            text = create_strings_randomly(2, allow_variable=True, count=1, let=False, num=True, sym=False, lang='en')
            text = "".join(text)
        elif random_num > 0.1:  # 纯字母
            text = create_strings_randomly(2, allow_variable=True, count=1, let=True, num=False, sym=False,
                                           lang='en')
            text = "".join(text)
        else:  # 数字+字母+符号
            text = create_strings_randomly(2, allow_variable=True, count=1, let=True, num=True, sym=True,
                                           lang='en')
            text = "".join(text)
        text = text.strip()  # 去除字符收尾两端的空格
        # 判断字符是否都在字体里
        text = valid_char(text, font)
        return text, font

    def aug_param_generate(self):
        """
        随机生成文本增强参数
        """
        self.gen_kwargs = dict(
            skewing_angle=np.random.randint(2),
            random_skew=np.random.randint(2),
            blur=1,
            random_blur=True,
            background_type=np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.1, 0.1, 0.3, 0.4]),
            distorsion_type=np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2]),  # 0: No 1: sin 2: cos
            distorsion_orientation=np.random.randint(2),  # 0: vertical 1: horizontal 2: both
            is_handwritten=False,
            name_format=0,
            width=-1,
            alignment=np.random.randint(3),  # 0: left 1: center 2: right, latter two should set width first
            text_color="#282828,#FFFFFF",
            orientation=np.random.choice([0, 1], p=[0.8, 0.2]),  # 0: horizontal 1: vertical
            space_width=round(np.random.rand() + 1, 1),  # random 1-2  the width of " " x factor
            character_spacing=np.random.randint(3),  # the width between characters
            margins=np.random.randint(1, 5, 4).tolist(),
            output_mask=False,
            fit=np.random.randint(2),  # 0: not tight the text 1: crop the text region
            word_split=False,
            stroke_width=np.random.randint(3),  # 笔画宽度
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
        while len(text) < 1:
            text, font = self.string_generate(self.fonts)

        # 有些字体在draw时会报错， allocation array size too large
        text_img = None
        while not text_img:
            text_img, text = FakeTextDataGenerator.generate(
                self.generated_count,
                text,
                font,
                None,
                self.img_h,
                None,
                image_dir=self.bg_image_dir,
                **self.gen_kwargs
            )
            if text_img:
                return (text_img, text)
            else:
                text, font = self.string_generate(self.fonts)
