# -*- coding: utf-8 -*-
# @Time : 2021/8/24 下午8:32
# @Author: yl
# @File: train.py
import os
import sys

sys.path.extend(['.', '..'])

from ocr.rec.model.recognizers import build_recognizer
from ocr.rec.model.losses import build_loss
from ocr.rec.core.evaluation import build_metric
from ocr.rec.model.converters import build_converter
from ocr.optimizer import build_optimizer
from ocr.rec.data import build_dataloader
from ocr.rec.apis.train import train_recognizer

import argparse
from .program import mergeConfig, loadConfig, loadCharDict

import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector.')
    parser.add_argument('--config', default='configs/rec/crnn/crnn_res50.yaml',
                        help='Train config file path.')
    parser.add_argument('--work-dir', help='The dir to save logs and models.')
    parser.add_argument('--device', default='0')
    parser.add_argument(
        '--load-from', help='The checkpoint file to load from.')
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
    cfg = loadConfig(args.pop('config'))
    debug = 1
    global_cfg = cfg["Global"]
    # cfg = mergeConfig(cfg, args)

    # get char idx dict
    char_dict_path = global_cfg['char_dict_path']
    char_idx_dict, idx_char_dict = loadCharDict(char_dict_path, global_cfg['character_type'],
                                                global_cfg['use_space_char'])

    num_classes = len(char_idx_dict.keys()) + 1
    cfg['Train']['dataset']['char_idx_dict'] = char_idx_dict
    cfg['Eval']['dataset']['char_idx_dict'] = char_idx_dict
    # build train data_loader
    train_data_loader = build_dataloader(cfg['Train'])

    # build eval data_loader
    if args['no_validate']:
        eval_data_loader = None
    else:
        eval_data_loader = build_dataloader(cfg["Eval"])

    # build model
    model_cfg = cfg['Model']
    # model_cfg['img_shape'] = cfg['Train']['dataset']['img_shape']
    model_cfg['decoder']['num_classes'] = num_classes

    model = build_recognizer(model_cfg)
    model = model()

    # build loss
    loss_fn = build_loss(cfg['Loss'])

    # build optimizer
    optimizer = build_optimizer(cfg["Optimizer"])

    # build converter
    cfg['Converter']['char_idx_dict'] = char_idx_dict
    converter = build_converter(cfg["Converter"])

    # build metric
    metrics = build_metric(cfg["Metric"])

    debug = 1
    train_recognizer(model, train_data_loader, eval_data_loader, loss_fn, optimizer, converter, metrics, cfg,
                     no_validate=args["no_validate"])


if __name__ == '__main__':
    main()
