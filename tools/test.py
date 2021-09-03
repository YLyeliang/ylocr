# -*- coding: utf-8 -*-
# @Time : 2021/9/1 下午4:00
# @Author: yl
# @File: test.py

import os
import sys

import cv2

sys.path.extend(['.', '..'])

from ocr.rec.core.evaluation import build_metric
from ocr.rec.model.converters import build_converter
from ocr.rec.apis.inference import initRecModel, inferenceRec
import argparse
from tools.program import mergeConfig, loadConfig, loadCharDict
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector.')
    parser.add_argument('--config', default='configs/rec/crnn/crnn_res34_ctc_mjsynth.yaml',
                        help='Train config file path.')
    parser.add_argument('--work-dir', help='The dir to save logs and models.')
    parser.add_argument('--device', default='0')
    parser.add_argument(
        '--load-from', default='work_dirs/crnn_res34_ctc_mjsynth/best_weights.hdf5',
        help='The checkpoint file to load from.')
    parser.add_argument(
        '--resume-from', help='The checkpoint file to resume from.')
    parser.add_argument(
        '--no-validate', default=False,
        action='store_true',
        help='Whether not to evaluate the checkpoint during training.')
    return parser.parse_args()


def main():
    args = parse_args().__dict__
    device = args.pop("device")
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.set_logical_device_configuration
    cfg = loadConfig(args.pop('config'))

    global_cfg = cfg["Global"]
    if args.get('load_from', False):
        cfg['Global']['load_from'] = args['load_from']

    # get char idx dict
    char_dict_path = global_cfg['char_dict_path']
    char_idx_dict, idx_char_dict = loadCharDict(char_dict_path, global_cfg['character_type'],
                                                global_cfg['use_space_char'])

    num_classes = len(char_idx_dict.keys()) + 1

    # build model
    cfg['Model']['decoder']['num_classes'] = num_classes

    model = initRecModel(cfg)

    # build converter
    cfg['Converter']['char_idx_dict'] = char_idx_dict
    cfg['Converter']['idx_char_dict'] = idx_char_dict
    converter = build_converter(cfg["Converter"])

    # build metric
    metrics = build_metric(cfg["Metric"])

    root = "../spark-ai-summit-2020-text-extraction/mjsynth_sample"
    files = os.listdir(root)
    for file in files:
        img_file = os.path.join(root, file)
        img = cv2.imread(img_file, 0)
        pred = inferenceRec([img, ], model, converter)


if __name__ == '__main__':
    main()
