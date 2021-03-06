# -*- coding: utf-8 -*-
# @Time : 2021/8/10 下午3:13
# @Author: yl
# @File: train.py
import codecs
import numpy as np
import re
import cv2

from ocr.rec.model.recognizers import build_recognizer
from ocr.rec.model.converters import build_converter
from ocr.rec.core.evaluation import build_metric
from ocr.rec.data.online_dataset import OnlineDataSet
import tensorflow as tf
import os
import time

recognizer_config = dict(
    model=dict(type='CRNNNet',
               backbone=dict(
                   type='ResNet',
               ),
               encoder=None,
               decoder=dict(
                   type='StepDecoder',
                   num_classes=None,
               ),
               loss=dict(
                   type='CTCLoss',
               )),
    converter=dict(
        type='CTCLabelConverter',
        char_idx_dict=None,
        character_type='cn',
        use_space_char=True
    ),
    metric=dict(
        type='RecMetric',
        keys=['line_acc', 'norm_edit_dis', 'char_acc']
    ),
    train_cfg=dict(
        batch_size=4,
    )
)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        pred, loss = model.forward_train(images, labels)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    return pred, loss


@tf.function
def val_step(images):
    pred = model(images)
    # y_true, label_length = labels['label'], labels['label_length']
    # val_loss = model.loss(pred, y_true, label_length)

    # val_loss(val_loss)
    return pred


def load_img_test(img_path, target_h):
    img_path = re.sub('\\\\', '/', img_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    target_w = int(w * target_h / h)
    img = cv2.resize(img, (target_w, target_h))
    img = img.astype(np.float)
    img = img / 255.0
    return img


def load_test_sample(img_root, label_root, char_idx_dict):
    label_list = os.listdir(label_root)
    sample_list = []
    for label_name in label_list:
        label_path = os.path.join(label_root, label_name)
        img_path = os.path.join(img_root, label_name[:-4] + ".jpg")
        try:
            with codecs.open(label_path, "rb", encoding='utf-8') as label_file:
                txt = label_file.readline()
                txt = txt.strip()
                flag = False
                for char in txt:
                    if char not in char_idx_dict:
                        flag = True
                        break
                if flag:
                    continue
                sample_list.append([img_path, txt])
        except:
            print("Error test sample:", label_name)
    return sample_list


def get_batch_test(sample_list, batch_size):
    """

    Args:
        sample_list: image path with text

    Returns:

    """
    x_data = []
    labels = []

    for i in range(len(sample_list)):
        x_data.append(load_img_test(sample_list[i], target_h=32))
        labels.append(sample_list[i])
        if len(x_data) == batch_size or i == len(sample_list) - 1:
            x_data = np.array(x_data)
            yield x_data, labels
            x_data = []
            labels = []


def print_val_result(pred_text, target_text):
    for i in range(len(pred_text)):
        print(pred_text[i], target_text[i])


# class IterBasedRunner():
#     def __init__(self):


if __name__ == '__main__':

    dict_path = "utils/ppocr_keys_v1.txt"
    dicts = []
    with open(dict_path, 'rb') as f:
        for p in f.readlines():
            p = p.decode('utf-8').strip("\n").strip("\r\n")
            dicts.append(p)
        dicts.append(" ")
    char_idx_dict = {p: i for i, p in enumerate(dicts)}
    model_cfg = recognizer_config['model']
    model_cfg['decoder']['num_classes'] = len(char_idx_dict.keys()) + 1
    batch_size = recognizer_config['train_cfg']['batch_size']
    recognizer_config['converter']['char_idx_dict'] = char_idx_dict

    strings = ["哈哈", 'Hhasf', "sdjflksdjlkjl", 'haha sfjslal']
    font_root = "trdg/fonts/"
    font_type = ['cn', 'latin']
    fonts = {}
    for type in font_type:
        ttfs = os.listdir(os.path.join(font_root, type))
        fonts_list = [os.path.join(font_root, type, ttf) for ttf in ttfs]
        fonts[type] = fonts_list
    dataset = OnlineDataSet(char_idx_dict, strings, max_sequence_len=32, fonts=fonts, bg_image_dir='data/crop_debug',
                            batch_size=4).next_train
    data_loader = tf.data.Dataset.from_generator(
        dataset,
        output_signature=(
            tf.TensorSpec(shape=(None, 32, 320, 3), name='train_data'),
            tf.TensorSpec(shape=(None, 32), dtype=tf.int64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64)
        )).map(lambda x, y, z: (x, {'label': y, "label_length": z}), num_parallel_calls=4).prefetch(buffer_size=8)

    model = build_recognizer(model_cfg)

    x = tf.keras.layers.Input(shape=(32, 320, 3), batch_size=4)

    converter = build_converter(recognizer_config['converter'])

    metric = build_metric(recognizer_config['metric'])

    lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=1000)

    optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True, clipvalue=5.)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')

    val_samples = load_test_sample()
    val_generator = get_batch_test(val_samples, batch_size=batch_size)

    eval_iter = 10
    s = time.time()


    for epoch in range(5):
        train_loss.reset_states()
        for iter, (images, labels) in enumerate(data_loader):

            pred, loss = train_step(images, labels)

            if (iter + 1) % eval_iter == 0:
                pred_text, target_text = converter(pred, labels['label'])
                metric_info = metric((pred_text, target_text))
                e = time.time()
                cost = (e - s) / 10
                s = time.time()

                template = f"Iter {iter + 1}, Cost: {cost:.2f} Loss: {train_loss.result()}, " \
                           f"Line Accuracy: {metric_info['line_acc']}, Char Accuracy: {metric_info['char_acc']}, " \
                           f"norm edit distance: {metric_info['norm_edit_dis']}"
                print(template)
            if iter % 20 == 0:
                metric.reset()
                # for val_images, val_labels in val_generator:
                #     pred = val_step(val_images)
                #     pred_text = converter(pred)
                #     metric((pred_text, (val_labels, 1)))
                #     print_val_result(pred_text[0])
                

            # model.save("my_model/best.pb", save_format="tf")
