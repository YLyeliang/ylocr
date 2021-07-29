import pandas as pd
from PIL import Image, ImageDraw
import os, sys, json
import cv2
import numpy as np
import math


def get_rotate_crop_image(img, points, direction="top", rotate_ratio=True):
    """
    with bbox points, rotate and crop the image into the rectangular image.
    Args:
        img:
        points(list[int]): coordinates in clock-wise. shape (4,2)
        direction(str): the direction of bottom of text region
        rotate_ratio(bool): Whether add the condition to rotate the text region

    Returns:

    """
    # 获取多边形的宽和高
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))

    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])

    # 透视变换
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE,
                                  flags=cv2.INTER_CUBIC)

    dst_img_height, dst_img_width = dst_img.shape[:2]
    # 宽比高大于1.5，则将其左旋90度，摆正送入识别网络
    if rotate_ratio:
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
    return dst_img


class Rotate(object):
    def __init__(self, image: Image.Image, coordinate):
        self.image = image.convert('RGB')
        self.coordinate = coordinate
        self.xy = [tuple(self.coordinate[k]) for k in ['left_top', 'right_top', 'right_bottom', 'left_bottom']]
        self._mask = None
        self.image.putalpha(self.mask)

    @property
    def mask(self):
        if not self._mask:
            mask = Image.new('L', self.image.size, 0)
            draw = ImageDraw.Draw(mask, 'L')
            draw.polygon(self.xy, fill=255)
            self._mask = mask
        return self._mask

    def run(self):
        image = self.rotation_angle()
        box = image.getbbox()
        return image.crop(box)

    def rotation_angle(self):
        x1, y1 = self.xy[0]
        x2, y2 = self.xy[1]
        angle = self.angle([x1, y1, x2, y2], [0, 0, 10, 0]) * -1
        return self.image.rotate(angle, expand=True)

    def angle(self, v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle


IMAGE_PATH = "../data/train/"
csvs = [
    "../data/Xeon1OCR_round1_train1_20210526.csv",
    "../data/Xeon1OCR_round1_train_20210524.csv",
    "../data/Xeon1OCR_round1_train2_20210526.csv"
]
train = pd.concat([pd.read_csv(c) for c in csvs])

idx = 0
for row in train.iloc[:].iterrows():
    path = json.loads(row[1]['原始数据'])['tfspath']
    img_path = IMAGE_PATH + path.split('/')[-1]
    flag = os.path.isfile(img_path)
    if not flag: continue
    labels = json.loads(row[1]['融合答案'])[0]

    image = Image.open(img_path)
    for label in labels[:]:
        text = json.loads(label['text'])['text']
        coord = [int(float(x)) for x in label['coord']]
        coordinate = {'left_top': coord[:2], 'right_top': coord[2:4], 'right_bottom': coord[4:6],
                      'left_bottom': coord[-2:]}
        rotate = Rotate(image, coordinate)
        corp_img = rotate.run().convert('RGB')
        corp_img.save(f'../data/det_images/{img_path.split("/")[-1][:-4]}_{idx}.jpg')

        with open('../data/train_list.txt', 'a+') as up:
            up.write(f'./det_images/{img_path.split("/")[-1][:-4]}_{idx}.jpg\t{text}\n')
        idx += 1

    print(path)
