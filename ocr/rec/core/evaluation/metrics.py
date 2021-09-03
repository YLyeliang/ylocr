# -*- coding: utf-8 -*-
# @Time : 2021/8/11 上午10:03
# @Author: yl
# @File: metrics.py
import tensorflow as tf
import Levenshtein


def line_acc(correct_line, all_line):
    return correct_line / all_line if all_line > 0 else 0


def char_acc(correct_char, all_char):
    return correct_char / all_char if all_char > 0 else 0


class RecMetric(object):
    """
    Calculate the Metrics of recognition, including Accuracy, norm distance.
    """

    def __init__(self, keys=['line_acc']):
        self.support_metrics = ['line_acc', 'char_acc', 'norm_edit_dis']
        self.metrics = {}
        self.keys = keys
        for key in keys:
            assert key in self.support_metrics, f"{key} is not supported, supported metrics are {self.support_metrics}"
            self.metrics[key] = 0.
        self.reset()

    def reset(self):
        self.correct_line = 0
        self.all_line = 0
        self.correct_char = 0
        self.all_char = 0
        self.norm_edit_dis = 0

    def __call__(self, pred_label, *args, **kwargs):
        """
        Calculate metrics
        Args:
            pred_label(list):list of pred and label, where each one are list(text,conf)
            *args:
            **kwargs:
        Returns:
        """
        preds, labels = pred_label
        correct_line = 0
        all_line = 0
        norm_edit_dis = 0.0
        all_char = 0
        correct_char = 0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")

            if 'char_acc' in self.keys:
                all_char += len(pred) + len(target)
                correct_char += (len(pred) + len(target)) * Levenshtein.ratio(pred, target)
            if "norm_edit_dis" in self.keys:
                norm_edit_dis += Levenshtein.distance(pred, target) / max(len(pred), len(target), 1)

            if 'line_acc' in self.keys:
                if pred == target:
                    correct_line += 1
                all_line += 1

        self.correct_line += correct_line
        self.correct_char += correct_char
        self.all_line += all_line
        self.all_char += all_char
        self.norm_edit_dis += norm_edit_dis
        self.metrics['line_acc'] = line_acc(correct_line, all_line)
        self.metrics['char_ac'] = char_acc(correct_char, all_char)
        self.metrics['norm_edit_dis'] = (1 - norm_edit_dis / all_line) if all_line > 0 else 0
        ret = {}
        for key in self.keys:
            ret[key] = self.metrics[key]
        return ret

    def get_metrics(self):
        """
        return metrics {
                 'line_acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        line_acc = 1.0 * self.correct_line / self.all_line if self.all_line else 0
        char_acc = 1.0 * self.correct_char / self.all_char if self.all_char else 0
        norm_edit_dis = 1 - self.norm_edit_dis / self.all_line if self.all_line else 0
        self.metrics['line_acc'] = line_acc
        self.metrics['char_acc'] = char_acc
        self.metrics['norm_edit_dis'] = norm_edit_dis
        ret = {}
        for key in self.keys:
            ret[key] = self.metrics[key]
        self.reset()
        return ret
