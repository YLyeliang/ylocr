# -*- coding: utf-8 -*-
# @Time : 2021/8/4 下午4:08
# @Author: yl
# @File: all_text.py
from trdg.apis.text_gen import textImgGen
from PIL import Image, ImageDraw, ImageFont

font = "trdg/fonts/latin/FFF_Tusj.ttf"
font = ImageFont.truetype(font=font, size=32)
img = Image.new("RGBA", (10, 2), (0, 0, 0, 0))

img_draw = ImageDraw.Draw(img)
img_draw.text((0, 15), 'H', fill=(0, 225, 225))
img.show()
debug = 1

if __name__ == '__main__':
    generator = textImgGen(batch_size=8, img_h=32, )
