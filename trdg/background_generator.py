import cv2
import math
import os
import random as rnd
import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageColor


def gaussian_noise(height, width):
    """
        Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add gaussian noise
    cv2.randn(image, 235, 10)

    return Image.fromarray(image).convert("RGBA")


def plain_white(height, width):
    """
        Create a plain white background
    """

    return Image.new("L", (width, height), 255).convert("RGBA")


def quasicrystal(height, width):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = rnd.random() * 30 + 20  # frequency
    phase = rnd.random() * 2 * math.pi  # phase
    rotation_count = rnd.randint(10, 20)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            pixels[kw, kh] = c  # grayscale
    return image.convert("RGBA")


def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))


def plain_color(height, width, bg_color):
    """
    Create a background with random plain color
    Args:
        height:
        width:
        t_color: the text color
    """
    image = (np.ones((height, width, 3)) * bg_color).astype(np.uint8)
    return Image.fromarray(image, mode='RGB').convert("RGBA")


def image(height, width, pic):
    """
        Create a background with a image
    """

    if pic.size[0] < width:
        pic = pic.resize(
            [width, int(pic.size[1] * (width / pic.size[0]))], Image.ANTIALIAS
        )
    if pic.size[1] < height:
        pic = pic.resize(
            [int(pic.size[0] * (height / pic.size[1])), height], Image.ANTIALIAS
        )

    if pic.size[0] == width:
        x = 0
    else:
        x = rnd.randint(0, pic.size[0] - width)
    if pic.size[1] == height:
        y = 0
    else:
        y = rnd.randint(0, pic.size[1] - height)

    return pic.crop((x, y, x + width, y + height))


def generate_color(color):
    colors = [ImageColor.getrgb(c) for c in color.split(",")]
    c1, c2 = colors[0], colors[-1]
    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )
    return fill
