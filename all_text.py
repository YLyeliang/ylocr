# -*- coding: utf-8 -*-
# @Time : 2021/8/4 下午4:08
# @Author: yl
# @File: all_text.py

# 测试生成数据的部分字体是否有错误


from trdg.apis.text_gen import textImgGen
from PIL import Image, ImageDraw, ImageFont
import os
fonts = os.listdir("trdg/fonts/latin/")
error_font =[]
for font_name in fonts:
# font = "trdg/fonts/latin/FFF_Tusj.ttf"
# font = "trdg/fonts/latin/Aller_Lt.ttf"
    font = os.path.join("trdg/fonts/latin/",font_name)
    font = ImageFont.truetype(font=font, size=32)
    img = Image.new("RGB", (264, 37), (0, 0, 0,0))

    img_draw = ImageDraw.Draw(img, mode='RGBA')
    # img_draw.fontmode = '1'
    try:
        img_draw.text((51, 0), 'H', fill=(232, 211, 89), font=font,stroke_width=1,stroke_fill=(221,78,123))
    except Exception as e:
        error_font.append(font_name)
    # img.show()
    debug = 1
print(error_font)
if __name__ == '__main__':
    generator = textImgGen(batch_size=8, img_h=32, )
