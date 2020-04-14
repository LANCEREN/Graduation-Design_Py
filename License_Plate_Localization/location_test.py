#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-19 10:29:34
#   Description :
#
#================================================================


import os, sys
import shutil
from pathlib2 import Path
from global_var import globalVars
import cv2
import numpy as np
import tensorflow as tf
import License_Plate_Localization.core.utils as utils
from License_Plate_Localization.core.config import cfg
from License_Plate_Localization.core.yolov3 import YOLOv3, decode


def TestModel():
    INPUT_SIZE   = 416
    NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES.__str__()))
    CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES.__str__())

    predicted_dir_path = globalVars.projectPath / Path('License_Plate_Localization', 'data', 'detection', 'mAP', 'predicted')
    ground_truth_dir_path = globalVars.projectPath / Path('License_Plate_Localization', 'data', 'detection', 'mAP', 'ground-truth')

    print(predicted_dir_path)
    if os.path.exists(predicted_dir_path.__str__()): shutil.rmtree(predicted_dir_path.__str__())
    if os.path.exists(ground_truth_dir_path.__str__()): shutil.rmtree(ground_truth_dir_path.__str__())
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH.__str__()): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH.__str__())

    os.mkdir(predicted_dir_path.__str__())
    os.mkdir(ground_truth_dir_path.__str__())
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH.__str__())

    # Build Model
    input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model_name = "yolov3"
    model_path = cfg.COMMON.MODEL_DIR_PATH / Path(model_name)
    model.load_weights(model_path.__str__())

    with open(cfg.TEST.ANNOT_PATH.__str__(), 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt=[]
                classes_gt=[]
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = ground_truth_dir_path / Path(f'{num}.txt')
            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path.__str__(), 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = predicted_dir_path / Path(f'{num}.txt')
            # Predict Process
            image_size = image.shape[:2]
            image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            pred_bbox = model.predict(image_data)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
            bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')


            if cfg.TEST.DECTECTED_IMAGE_PATH.__str__() is not None:
                image = utils.draw_bbox(image, bboxes)
                detected_image_path = cfg.TEST.DECTECTED_IMAGE_PATH / Path(image_name)
                cv2.imwrite(detected_image_path.__str__(), image)

            with open(predict_result_path.__str__(), 'w') as f:
                for bbox in bboxes:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())

