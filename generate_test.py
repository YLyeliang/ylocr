# -*- coding: utf-8 -*-
# @Time : 2021/8/3 上午9:31
# @Author: yl
# @File: generate_test.py
import os
import numpy as np
from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)
import matplotlib.pyplot as plt

# The generators use the same arguments as the CLI, only as parameters
generator = GeneratorFromStrings(
    ['损失函数[公式]是在水平轴和上的曲面，因此曲面的', 'Test2 abc haha', 'Test3'],
    blur=1,
    random_blur=True, language='cn', image_dir="trdg/images", background_type=3, orientation=0,
    distorsion_type=1, distorsion_orientation=0, word_split=True, skewing_angle=2, random_skew=True, fit=True,
    margins=(0, 0, 0, 0)
)

# from fontTools.ttLib import TTFont
#
# font = TTFont("trdg/fonts/cn/SourceHanSans-Normal.ttf")
# uniMap = font['cmap'].tables[0].ttFont.getBestCmap()

for img, lbl in generator:
    # Do something with the pillow images here.
    plt.figure()
    plt.imshow(img)
    print(img.size)
    print(np.asarray(img).shape)
    # plt.show()
    # plt.imshow(img[1])
    plt.show()
    debug = 1
