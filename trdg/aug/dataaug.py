# -*- coding: utf-8 -*-
# @Time : 2021/8/10 上午9:33
# @Author: yl
# @File: dataaug.py
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np


def img_aug(image):
    """

    Args:
        image: Image with RGB format

    Returns:

    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    sometimes_5 = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes_1 = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes_8 = lambda aug: iaa.Sometimes(0.8, aug)
    sometimes_2 = lambda aug: iaa.Sometimes(0.2, aug)

    augmentations = iaa.Sequential([
        sometimes_2(iaa.CropAndPad(percent=(-0.02, 0.02), pad_mode='edge', pad_cval=(0, 255))),
        iaa.Sequential([iaa.size.Resize(0.6), iaa.size.Resize(1 / 0.6)]),

        # inverts the text ( make black into white)
        sometimes_1(iaa.Invert(1, per_channel=True)),

        # Affine
        sometimes_2(iaa.Affine(
            scale={"x": (0.8, 1), "y": (0.8, 1)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0, 0), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
            rotate=(-2, 2),  # rotate by -45 to +45
            shear=(-2, 2),  # shear by -16 + 16
            order=[0, 1],  # use nearest neighbour or bilinear
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=["edge"]
        )),
        # change brightness of images ( by -10 to 10 of original value)
        sometimes_2(iaa.Add((-10, 10,), per_channel=0.5)),
        sometimes_8(iaa.AddToHueAndSaturation((-200, 200))),  # hue and saturation
        # sometimes_5(iaa.contrast.LinearContrast((0.8, 5), per_channel=0.5)),  # improve or worsen the contrast
        sometimes_1(iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.2)),
    ])
    aug_img = augmentations(image=image)
    return Image.fromarray(aug_img)
