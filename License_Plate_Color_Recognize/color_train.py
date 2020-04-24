import os, shutil

import cv2
import numpy as np
from pathlib2 import Path
from License_Plate_Color_Recognize.core.config import cfg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


# K.set_image_dim_ordering('tf')


class Train:
    plateType = ["蓝牌", "单层黄牌", "新能源车牌", "白色", "黑色-港澳"]

    def Getmodel_tensorflow(self, nb_classes):
        # nb_classes = len(charset)

        img_rows, img_cols = 9, 34
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        nb_pool = 2
        # convolution kernel size
        nb_conv = 3

        # x = np.load('x.npy')
        # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
        # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
        # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

        model = Sequential()
        model.add(Conv2D(16, (5, 5), input_shape=(img_rows, img_cols, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def SimplePredict(self, img):

        model = self.Getmodel_tensorflow(5)
        model_name = "plate_type.h5"
        model_path = cfg.COMMON.MODEL_DIR_PATH / Path(model_name)
        model.load_weights(model_path.__str__())
        model.save(model_path.__str__())

        image = cv2.resize(img, (34, 9))
        image = image.astype(np.float) / 255
        res = np.array(model.predict(np.array([image]))[0])
        return res.argmax()


if __name__ == "__main__":
    pass