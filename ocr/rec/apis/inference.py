# -*- coding: utf-8 -*-
# @Time : 2021/9/1 下午4:03
# @Author: yl
# @File: inference.py
import cv2
import numpy as np

from ocr.rec.model.recognizers import build_recognizer


def initRecModel(cfg):
    model_cfg = cfg['Model']
    global_cfg = cfg['Global']
    model = build_recognizer(model_cfg)
    model = model()
    if global_cfg['load_from']:
        model.load_weights(global_cfg['load_from'])
        print(f"load from {global_cfg['load_from']}")
    return model


def inferenceRec(imgs, model, converter):
    """
    Args:
        imgs:
    Returns:
    """
    data = []
    for img in imgs:
        img = img.astype(np.float32) / 255.
        h, w = img.shape[:2]
        ratio = 32 / h
        w = int(w * ratio)
        img = cv2.resize(img, (w, 32))
        img = img[:, :, np.newaxis]
        data.append(img)
    data = np.stack(data)
    pred = model(data, training=False)
    pred_text = converter(pred, None)
    # metric((pred_text, target_text))
    return pred_text
