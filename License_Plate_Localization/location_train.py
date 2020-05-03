#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
# ================================================================


import os, sys
import shutil
from pathlib2 import Path
from global_var import globalVars
import cv2
import numpy as np
import tensorflow as tf
import License_Plate_Localization.core.utils as utils
from License_Plate_Localization.core.config import cfg
from License_Plate_Localization.core.dataset import Dataset
from License_Plate_Localization.core.yolov3 import YOLOv3, decode, compute_loss


class Train():
    def TrainModel(self, mode='new'):

        trainset = Dataset('train')
        steps_per_epoch = len(trainset)
        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
        total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

        input_tensor = tf.keras.layers.Input([416, 416, 3])
        conv_tensors = YOLOv3(input_tensor)

        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = decode(conv_tensor, i)
            output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)

        model = tf.keras.Model(input_tensor, output_tensors)
        model_name = "yolov3"
        model_path = cfg.COMMON.MODEL_DIR_PATH / Path(model_name)
        if mode == "continue":
            model.load_weights(model_path.__str__())

        optimizer = tf.keras.optimizers.Adam()
        if os.path.exists(cfg.COMMON.LOG_DIR_PATH.__str__()): shutil.rmtree(cfg.COMMON.LOG_DIR_PATH.__str__())
        writer = tf.summary.create_file_writer(cfg.COMMON.LOG_DIR_PATH.__str__())

        def train_step(image_data, target):
            with tf.GradientTape() as tape:
                pred_result = model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                for i in range(3):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = compute_loss(pred, conv, *target[i], i)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                total_loss = giou_loss + conf_loss + prob_loss

                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                         "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                                   giou_loss, conf_loss,
                                                                   prob_loss, total_loss))
                # update learning rate
                global_steps.assign_add(1)
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
                else:
                    lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                        (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                    )
                optimizer.lr.assign(lr.numpy())

                # writing summary data
                with writer.as_default():
                    tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                    tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                    tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                    tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                    tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                writer.flush()

        for epoch in range(cfg.TRAIN.EPOCHS):
            for image_data, target in trainset:
                train_step(image_data, target)
            model.save_weights(model_path.__str__())

    def predictImg(self, img, image_name):
        INPUT_SIZE = 416
        NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES.__str__()))
        CLASSES = utils.read_class_names(cfg.YOLO.CLASSES.__str__())

        if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH.__str__()): shutil.rmtree(
            cfg.TEST.DECTECTED_IMAGE_PATH.__str__())
        os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH.__str__())

        # Build Model
        input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
        feature_maps = YOLOv3(input_layer)

        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, i)
            bbox_tensors.append(bbox_tensor)

        model = tf.keras.Model(input_layer, bbox_tensors)
        model_name = "yolov3"
        model_path = cfg.COMMON.MODEL_DIR_PATH / Path(model_name)
        model.load_weights(model_path.__str__())

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
        bboxes = sorted(bboxes, key=lambda x: x[-2])

        if cfg.TEST.DECTECTED_IMAGE_PATH.__str__() is not None:
            image = utils.draw_bbox(image, bboxes)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            plate_whole = image
            # detected_image_path = cfg.TEST.DECTECTED_IMAGE_PATH / Path(f"{image_name}.jpg")

        try:
            if len(bboxes) == 0:
                raise Exception("未检测到车牌")
        except Exception as err:
            print("Lpl error: ", err)
            raise
        else:
            coor = np.array(bboxes[-1][:4], dtype=np.int32)
        plate_precise = img[coor[1]:coor[3], coor[0]:coor[2]]

        if coor[3] - coor[1] < image_size[0] // 2 and coor[2] - coor[0] < image_size[1] // 2:
            if image_size[0] // 4 < (coor[1] + coor[3]) // 2 < 3 * image_size[0] // 4 \
                    and image_size[1] // 4 < (coor[0] + coor[2]) // 2 < 3 * image_size[1] // 4:
                plate_general = image[
                                (coor[1] + coor[3]) // 2 - image_size[0] // 4: (coor[1] + coor[3]) // 2 + image_size[
                                    0] // 4,
                                (coor[0] + coor[2]) // 2 - image_size[1] // 4: (coor[0] + coor[2]) // 2 + image_size[
                                    1] // 4]
            else:
                if (coor[1] + coor[3]) // 2 < image_size[0] // 4:
                    plate_general = image[0: image_size[0] // 2, :]
                else:
                    plate_general = image[image_size[0] // 2: image_size[0], :]
                if (coor[0] + coor[2]) // 2 < image_size[1] // 4:
                    plate_general = plate_general[:, 0: image_size[1] // 2]
                else:
                    plate_general = plate_general[:, image_size[1] // 2: image_size[1]]
        else:
            plate_general = image
        plateConf = bboxes[-1][4]
        # cv2.imshow("x", plate_general)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return plate_whole, plate_general, plate_precise, plateConf


if __name__ == "__main__":
    pass
