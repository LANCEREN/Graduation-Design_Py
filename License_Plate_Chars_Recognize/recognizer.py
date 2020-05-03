from License_Plate_Chars_Recognize.core.config import cfg
import cv2
import numpy as np
from pathlib2 import Path
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


# K.set_image_dim_ordering('tf')


class RecognizeTrain:

    def __init__(self):
        self.index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11,
                      "皖": 12,
                      "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23,
                      "云": 24,
                      "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35,
                      "5": 36,
                      "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47,
                      "H": 48,
                      "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59,
                      "V": 60,
                      "W": 61, "X": 62, "Y": 63, "Z": 64, "港": 65, "学": 66, "O": 67, "使": 68, "警": 69, "澳": 70,
                      "挂": 71};

        self.chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
                      "粤", "桂",
                      "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8",
                      "9", "A",
                      "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
                      "Q", "R", "S", "T", "U", "V", "W", "X",
                      "Y", "Z", "港", "学", "O", "使", "警", "澳", "挂"];

        # 构建网络
        self.model = self.Getmodel_tensorflow(65)
        self.model_name = "char_rec.h5"
        self.model_path = Path(cfg.COMMON.MODEL_DIR_PATH, self.model_name)
        self.model.load_weights(self.model_path.__str__())
        # model.save("./model/char_rec.h5")

        self.model_ch = self.Getmodel_ch(31)
        self.model_ch_name = "char_chi_sim.h5"
        self.model_ch_path = Path(cfg.COMMON.MODEL_DIR_PATH, self.model_ch_name)
        self.model_ch.load_weights(self.model_ch_path.__str__())
        # model_ch.save_weights("./model/char_chi_sim.h5")

    def Getmodel_tensorflow(self, nb_classes):
        # nb_classes = len(charset)

        img_rows, img_cols = 23, 23
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
        model.add(Conv2D(32, (5, 5), input_shape=(img_rows, img_cols, 1)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def Getmodel_ch(self, nb_classes):
        # nb_classes = len(charset)

        img_rows, img_cols = 23, 23
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
        model.add(Conv2D(32, (5, 5), input_shape=(img_rows, img_cols, 1)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(756))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def SimplePredict(self, image, pos):
        image = cv2.resize(image, (23, 23))
        image = cv2.equalizeHist(image)
        image = image.astype(np.float) / 255
        image -= image.mean()
        # image = np.expand_dims(image, 3)
        image = np.expand_dims(image, 2)
        if pos != 0:
            res = np.array(self.model.predict(np.array([image]))[0])
        else:
            res = np.array(self.model_ch.predict(np.array([image]))[0])

        zero_add = 0;

        if pos == 0:
            res = res[:31]
        elif pos == 1:
            res = res[31 + 10:65]
            zero_add = 31 + 10
        else:
            res = res[31:]
            zero_add = 31

        max_id = res.argmax()

        return res.max(), self.chars[max_id + zero_add], max_id + zero_add

    def PredictPlate(self, refined):
        plateString = ""
        totalConfidence = 0.00
        for i, section in enumerate(refined):
            res_pre = self.SimplePredict(section, i)
            totalConfidence += res_pre[0]
            plateString += res_pre[1]
        averageConfidence = totalConfidence / len(plateString)
        return plateString, averageConfidence


class LPR():

    def __init__(self):
        self.chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁",
                      u"豫", u"鄂", u"湘", u"粤", u"桂",
                      u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5",
                      u"6", u"7", u"8", u"9", u"A",
                      u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S",
                      u"T", u"U", u"V", u"W", u"X",
                      u"Y", u"Z", u"港", u"学", u"使", u"警", u"澳", u"挂", u"军", u"北", u"南", u"广", u"沈", u"兰", u"成", u"济",
                      u"海", u"民", u"航", u"空"
                      ]
        self.watch_cascade_model_name = "cascade.xml"
        self.watch_cascade_model_path = Path(cfg.COMMON.MODEL_DIR_PATH, self.watch_cascade_model_name)
        self.watch_cascade = cv2.CascadeClassifier(self.watch_cascade_model_path.__str__())
        self.finemapping_model_name = "model12.h5"
        self.finemapping_model_path = Path(cfg.COMMON.MODEL_DIR_PATH, self.finemapping_model_name)
        self.modelFineMapping = self.model_finemapping(self.finemapping_model_path.__str__())
        self.seqRec_model_name = "ocr_plate_all_gru.h5"
        self.seqRec_model_path = Path(cfg.COMMON.MODEL_DIR_PATH, self.seqRec_model_name)
        self.modelSeqRec = self.model_seq_rec(self.seqRec_model_path.__str__())

    def computeSafeRegion(self, shape, bounding_rect):
        top = bounding_rect[1]  # y
        bottom = bounding_rect[1] + bounding_rect[3]  # y +  h
        left = bounding_rect[0]  # x
        right = bounding_rect[0] + bounding_rect[2]  # x +  w
        min_top = 0
        max_bottom = shape[0]
        min_left = 0
        max_right = shape[1]
        if top < min_top:
            top = min_top
        if left < min_left:
            left = min_left
        if bottom > max_bottom:
            bottom = max_bottom
        if right > max_right:
            right = max_right
        return [left, top, right - left, bottom - top]

    def cropImage(self, image, rect):
        x, y, w, h = self.computeSafeRegion(image.shape, rect)
        return image[y:y + h, x:x + w]

    def detectPlateRough(self, image_gray, resize_h=720, en_scale=1.08, top_bottom_padding_rate=0.05):
        if top_bottom_padding_rate > 0.2:
            print("error:top_bottom_padding_rate > 0.2:", top_bottom_padding_rate)
            exit(1)
        height = image_gray.shape[0]
        padding = int(height * top_bottom_padding_rate)
        scale = image_gray.shape[1] / float(image_gray.shape[0])
        image = cv2.resize(image_gray, (int(scale * resize_h), resize_h))
        image_color_cropped = image[padding:resize_h - padding, 0:image_gray.shape[1]]
        image_gray = cv2.cvtColor(image_color_cropped, cv2.COLOR_RGB2GRAY)
        watches = self.watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),
                                                      maxSize=(36 * 40, 9 * 40))
        cropped_images = []
        for (x, y, w, h) in watches:
            x -= w * 0.14
            w += w * 0.28
            y -= h * 0.15
            h += h * 0.3
            cropped = self.cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
            cropped_images.append([cropped, [x, y + padding, w, h]])
        return cropped_images

    def fastdecode(self, y_pred):
        results = ""
        confidence = 0.0
        table_pred = y_pred.reshape(-1, len(self.chars) + 1)
        res = table_pred.argmax(axis=1)
        for i, one in enumerate(res):
            if one < len(self.chars) and (i == 0 or (one != res[i - 1])):
                results += self.chars[one]
                confidence += table_pred[i][one]
        confidence /= len(results)
        return results, confidence

    def model_seq_rec(self, model_path):
        width, height, n_len, n_class = 164, 48, 7, len(self.chars) + 1
        rnn_size = 256
        input_tensor = Input((164, 48, 3))
        x = input_tensor
        base_conv = 32
        for i in range(3):
            x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        conv_shape = x.get_shape()
        x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1', reset_after=False)(x)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b',
                     reset_after=False)(x)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2', reset_after=False)(
            gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b',
                     reset_after=False)(gru1_merged)
        x = concatenate([gru_2, gru_2b])
        x = Dropout(0.25)(x)
        x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
        base_model = Model(inputs=input_tensor, outputs=x)
        base_model.load_weights(model_path)
        return base_model

    def model_finemapping(self, model_path):
        input = Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = Activation("relu", name='relu1')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = Activation("relu", name='relu2')(x)
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = Activation("relu", name='relu3')(x)
        x = Flatten()(x)
        output = Dense(2, name="dense")(x)
        output = Activation("relu", name='relu4')(output)
        model = Model([input], [output])
        model.load_weights(model_path)
        return model

    def finemappingVertical(self, image, rect):
        resized = cv2.resize(image, (66, 16))
        resized = resized.astype(np.float) / 255
        res_raw = self.modelFineMapping.predict(np.array([resized]))[0]
        res = res_raw * image.shape[1]
        res = res.astype(np.int)
        H, T = res
        H -= 3
        if H < 0:
            H = 0
        T += 2;
        if T >= image.shape[1] - 1:
            T = image.shape[1] - 1
        rect[2] -= rect[2] * (1 - res_raw[1] + res_raw[0])
        rect[0] += res[0]
        image = image[:, H:T + 2]
        image = cv2.resize(image, (int(136), int(36)))
        return image, rect

    def recognizeOne(self, src):
        x_tempx = src
        x_temp = cv2.resize(x_tempx, (164, 48))
        x_temp = x_temp.transpose(1, 0, 2)
        y_pred = self.modelSeqRec.predict(np.array([x_temp]))
        y_pred = y_pred[:, 2:, :]
        return self.fastdecode(y_pred)

    def SimpleRecognizePlateByE2E(self, image):
        images = self.detectPlateRough(image, image.shape[0], top_bottom_padding_rate=0.1)
        res_set = []
        for j, plate in enumerate(images):
            plate, rect = plate
            image_rgb, rect_refine = self.finemappingVertical(plate, rect)
            res, confidence = self.recognizeOne(image_rgb)
            res_set.append([res, confidence, rect_refine])
        return res_set

    def RecognizePlate(self, plateImage):
        plateString, confidence = self.recognizeOne(plateImage)
        return plateString, confidence
