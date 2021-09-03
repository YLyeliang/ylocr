# -*- coding: utf-8 -*-
# @Time : 2021/8/26 下午4:55
# @Author: yl
# @File: train.py

import logging
import os
import time

import tensorflow as tf


@tf.function
def train_step(inputs, model, loss, optimizer):
    with tf.GradientTape() as tape:
        pred = model(inputs["input"], training=True)
        loss_value = loss(inputs, pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return pred, loss_value


# @tf.function
def test_step(inputs, model, loss):
    pred = model(inputs["input"], training=False)
    loss_value = loss(inputs, pred)
    return pred, loss_value


def train_recognizer(model, train_data_loader, val_data_loader,
                     loss_fn, optimizer, lr_scheduler, converter, metric, config, no_validate, logger):
    global_cfg = config["Global"]
    cal_metric_during_train = global_cfg.get('cal_metric_during_train',
                                             False)
    epoch_num = global_cfg['epoch_num']
    log_iter_step = global_cfg['log_iter_step']
    eval_iter_step = global_cfg['eval_iter_step']
    save_epoch_step = global_cfg['save_epoch_step']

    save_root = global_cfg['work_dir']
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # model.compile(optimizer=optimizer)
    model.summary()

    global_iter = 0
    best_line_acc = 0

    data_length = config['train_length']

    for epoch in range(epoch_num):

        metric.reset()
        data_st = time.time()
        inf_st = time.time()
        for iter, inputs in enumerate(train_data_loader):

            data_et = time.time()

            pred, loss_value = train_step(inputs, model, loss_fn, optimizer)

            lr_value = lr_scheduler(global_iter)
            optimizer.learning_rate = lr_value

            inf_et = time.time()
            inf_cost = inf_et - inf_st

            if (global_iter + 1) % log_iter_step == 0:
                pred_text, target_text = converter(pred, inputs['label'])

                if cal_metric_during_train:
                    metric((pred_text, target_text))
                    metric_info = metric.get_metrics()

                    metric_str = ""
                    for key, val in metric_info.items():
                        tmp_str = f"{key}: {val:0.4f}. "
                        metric_str += tmp_str

                # Time cost
                data_cost = data_et - data_st
                eta = inf_cost * (epoch_num * data_length - global_iter) / 60

                logger.info(
                    f"Epoch: {epoch + 1}/{epoch_num}. Iter: {global_iter + 1}. ETA: {eta:.2f} min. "
                    f"Loss: {loss_value.numpy():.2f}. Lr: {lr_value:.6f}. {metric_str if cal_metric_during_train else ''}"
                    f"data:{data_cost:.4f}. train: {inf_cost:.4f}. sample: {data_length}")

            if not no_validate and (global_iter + 1) % eval_iter_step == 0:
                metric.reset()
                total_val_loss = 0
                for val_iter, val_inputs in enumerate(val_data_loader):
                    pred, loss_value = test_step(val_inputs, model, loss_fn)
                    pred_text, target_text = converter(pred, val_inputs['label'])
                    metric((pred_text, target_text))
                    total_val_loss += loss_value.numpy()

                metric_info = metric.get_metrics()
                metric_str = ""
                for key, val in metric_info.items():
                    tmp_str = f"{key}: {val:0.4f}. "
                    metric_str += tmp_str
                logger.info(
                    f"Eval: Loss: {total_val_loss / (val_iter + 1)}. {metric_str}")

            data_st = time.time()
            inf_st = time.time()
            global_iter += 1

        if (epoch + 1) % save_epoch_step == 0:
            # model.save(global_cfg['work_dir'])
            if not no_validate:
                metric.reset()
                total_val_loss = 0
                for val_iter, val_inputs in enumerate(val_data_loader):
                    pred, loss_value = test_step(val_inputs, model, loss_fn)
                    pred_text, target_text = converter(pred, val_inputs['label'])
                    metric((pred_text, target_text))
                    total_val_loss += loss_value.numpy()

                metric_info = metric.get_metrics()
                metric_str = ""
                for key, val in metric_info.items():
                    tmp_str = f"{key}: {val:0.4f}. "
                    metric_str += tmp_str
                logger.info(
                    f"Eval: Loss: {total_val_loss / (val_iter + 1)}. {metric_str}")

                best_weights = os.path.join(save_root, "best_weights.hdf5")
                best_model = os.path.join(save_root, "best_model.hdf5")

                line_acc = metric_info['line_acc']
                if line_acc >= best_line_acc:
                    best_line_acc = line_acc
                    model.save_weights(best_weights)
                    model.save(best_model)
                    logger.info(f"Save best model to: {best_weights}, {best_model}, epoch: {epoch + 1}")

            save_weights = os.path.join(save_root, f"last_weights.hdf5")
            save_model = os.path.join(save_root, f"last_model.hdf5")

            model.save_weights(save_weights)
            model.save(save_model)
