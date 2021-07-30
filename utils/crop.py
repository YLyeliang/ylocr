import pandas as pd
from PIL import Image, ImageDraw
import os, sys, json
import cv2
import numpy as np
import math
from urllib.parse import unquote


def intersection(bboxA, bboxB):
    """
    Calculate the intersection between bboxes.
    Examples:
        >>> bboxA = np.array([[10, 10, 20, 20], [20, 20, 30, 30], [25, 25, 35, 35]])
        >>> bboxB = np.array([[30, 30, 40, 40]])
        >>> intersection(bboxA,bboxB)
        >>> [[0],[0],[25]]
    Args:
        bboxA(np.ndarray):(N,4)
        bboxB: (M,4)
    Returns:
        inter(np.ndarray): with shape (N X M), which is a intersection mapping between bboxes.
    """
    bottom_right = np.minimum(bboxA[:, None, 2:], bboxB[:, 2:])
    top_left = np.maximum(bboxA[:, None, :2], bboxB[:, :2])
    inter = (bottom_right - top_left).clip(0).prod(2)
    return inter


def random_bg_region(img_shape, h_ratio=0.2, w_ratio=0.8):
    """
    random generate the left top point and bottom right point in the image
    Args:
        img_shape(list): h,w

    Returns:
        list of xyxy
    """
    h, w = img_shape
    max_h = max(h * h_ratio, 33)
    max_w = max(w * w_ratio, 101)
    crop_h = np.random.randint(32, max_h)
    crop_w = np.random.randint(100, max_w)
    x_min = np.random.randint(0, w - crop_w)
    y_min = np.random.randint(0, h - crop_h)
    x_max = x_min + crop_w
    y_max = y_min + crop_h
    return [x_min, y_min, x_max, y_max]


def crop_bg_exclude_text(image, bboxes, num=10):
    """
    Crop a image into multiple small regions, while these
    Args:
        image:
        bboxes(list(np.ndarray): a list of bbox, each one is xyxy
        num:
    Returns:
    """
    bboxes = np.array(bboxes)
    shape = image.shape[:2]
    patches = []
    for i in range(num):
        crop = random_bg_region(shape)
        inter = intersection(bboxes, np.array(crop)[np.newaxis, :])
        overlap = (inter > 0).any()  # if crop region overlapped the text region
        if overlap:
            continue
        else:
            patch = image[crop[1]:crop[3], crop[0]:crop[2]]
            patches.append(patch)
    return patches


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


if __name__ == '__main__':
    out_root = "../data/crop_debug"
    debug = 1
    IMAGE_PATH = "../data/train/"
    csvs = [
        # "../data/Xeon1OCR_round1_train1_20210526.csv",
        # "../data/Xeon1OCR_round1_train_20210524.csv",
        "../data/Xeon1OCR_round1_train2_20210526.csv"
    ]

    train = pd.concat([pd.read_csv(c) for c in csvs])
    for i, row in enumerate(train.iloc[:].iterrows()):
        path = json.loads(row[1]['原始数据'])['tfspath']
        labels = json.loads(row[1]['融合答案'])[0]
        img_name = path.split('/')[-1]
        img_name = unquote(img_name)
        img_path = IMAGE_PATH + img_name
        if not os.path.isfile(img_path):
            print("not file")
            continue
        image = cv2.imread(img_path)
        shape = image.shape[:2]
        if shape[0] < 50 or shape[1] < 128:
            continue
        points = []
        for label in labels[:]:
            coord = [int(float(x)) for x in label['coord']]
            coord = np.array(coord).reshape(4, 2)
            x_min = np.min(coord[:, 0])
            y_min = np.min(coord[:, 1])
            x_max = np.max(coord[:, 0])
            y_max = np.max(coord[:, 1])
            points.append([x_min, y_min, x_max, y_max])
        if len(points) == 0:
            continue
        points = np.array(points)
        patches = crop_bg_exclude_text(image, points, num=5)
        for num, patch in enumerate(patches):
            patch_name = img_name[:-4] + f"_{num}.jpg"

            cv2.imwrite(os.path.join(out_root, patch_name), patch)
        print(f"{i}:    {len(patches)}")

# original crop text region into patches
# IMAGE_PATH = "../data/train/"
# csvs = [
#     # "../data/Xeon1OCR_round1_train1_20210526.csv",
#     "../data/Xeon1OCR_round1_train_20210524.csv",
#     "../data/Xeon1OCR_round1_train2_20210526.csv"
# ]
# train = pd.concat([pd.read_csv(c) for c in csvs])
#
# idx = 0
# for row in train.iloc[:].iterrows():
#     path = json.loads(row[1]['原始数据'])['tfspath']
#     img_path = IMAGE_PATH + path.split('/')[-1]
#     flag = os.path.isfile(img_path)
#     if not flag: continue
#     labels = json.loads(row[1]['融合答案'])[0]
#
#     image = Image.open(img_path)
#     for label in labels[:]:
#         text = json.loads(label['text'])['text']
#         coord = [int(float(x)) for x in label['coord']]
#         coordinate = {'left_top': coord[:2], 'right_top': coord[2:4], 'right_bottom': coord[4:6],
#                       'left_bottom': coord[-2:]}
#         rotate = Rotate(image, coordinate)
#         corp_img = rotate.run().convert('RGB')
#         corp_img.save(f'../data/det_images/{img_path.split("/")[-1][:-4]}_{idx}.jpg')
#
#         with open('../data/train_list.txt', 'a+') as up:
#             up.write(f'./det_images/{img_path.split("/")[-1][:-4]}_{idx}.jpg\t{text}\n')
#         idx += 1
