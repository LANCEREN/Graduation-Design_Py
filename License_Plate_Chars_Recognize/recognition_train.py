import os
import shutil
import time
from pathlib2 import Path
from global_var import globalVars
import PIL
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Train():
    # 训练器的父类
    def __init__(self, trainTargetNumber, modelSaveDir, logDir, trainDatasetDir, validDatasetDir, trainTarget,
                 WIDTH_Column= 32,
                 HEIGHT_Row = 40,
                 CHANNEL = 1,
                 batch_size = 60,
                 iterations = 10,
                 validation_split = 0.1,
                 verbose = 1,
                 modelName = "model.h5"
                 ):

        # 加载数据 进行训练的准备工作
        time_begin = time.time()

        self.WIDTH_Column = WIDTH_Column
        self.HEIGHT_Row = HEIGHT_Row
        self.SIZE = HEIGHT_Row * WIDTH_Column
        self.CHANNEL = CHANNEL
        self.inputFormat = [-1, HEIGHT_Row, WIDTH_Column, CHANNEL]
        self.trainTarget = trainTarget
        self.trainTargetNumber = trainTargetNumber

        self.batch_size = batch_size
        self.iterations = iterations
        self.verbose = verbose
        self.validation_split = validation_split
        self.modelName = modelName
        self.modelSaveDir = modelSaveDir
        self.modelSavePath = self.modelSaveDir / Path(self.modelName)
        self.logDir = logDir

        self.trainDatasetDir = trainDatasetDir
        self.validDatasetDir =  validDatasetDir
        self.train_count = 0
        self.val_count = 0

        #生成缺省model
        self.defaultModel = self.Default_Model_Generator()

    def LoadData(self):
        # 获取训练和评估图片总数
        for i in range(0, self.trainTargetNumber):
            dir = self.trainDatasetDir / Path(f'{i}')
            for rt, dirs, files in os.walk(dir.__str__()):
                for filename in files:
                    self.train_count += 1
        for i in range(0, self.trainTargetNumber):
            dir = self.validDatasetDir / Path(f'{i}')
            for rt, dirs, files in os.walk(dir.__str__()):
                for filename in files:
                    self.val_count += 1

        # 定义对应维数和各维长度的数组
        self.train_images = np.zeros((self.train_count, self.HEIGHT_Row, self.WIDTH_Column, self.CHANNEL), dtype=int)
        self.train_labels = np.zeros((self.train_count, self.trainTargetNumber), dtype=int)
        # 定义对应维数和各维长度的数组
        self.val_images = np.zeros((self.val_count, self.HEIGHT_Row, self.WIDTH_Column, self.CHANNEL), dtype=int)
        self.val_labels = np.zeros((self.val_count, self.trainTargetNumber), dtype=int)

        print(f"用于train的照片shape为{self.train_images.shape},"
              f"用于train的标签shape为{self.train_labels.shape},"
              f"用于valid的照片shape为{self.val_images.shape},"
              f"用于valid的标签shape为{self.val_labels.shape}!")

        # 生成训练图片数据和标签
        index = 0
        for i in range(0, self.trainTargetNumber):
            dir = self.trainDatasetDir / Path(f'{i}')
            for rt, dirs, files in os.walk(dir.__str__()):
                for filename in files:
                    fullFileName = dir / Path(filename)
                    img = PIL.Image.open(fullFileName.__str__())
                    self.train_images[index] = np.array(img).reshape(self.inputFormat)
                    self.train_labels[index][i] = 1
                    index += 1

        # 生成评估测试的图片数据和标签
        index = 0
        for i in range(0, self.trainTargetNumber):
            dir = self.validDatasetDir / Path(f'{i}')
            for rt, dirs, files in os.walk(dir.__str__()):
                for filename in files:
                    fullFileName = dir / Path(filename)
                    img = PIL.Image.open(fullFileName.__str__())
                    self.val_images[index] = np.array(img).reshape(self.inputFormat)
                    self.val_labels[index][i] = 1
                    index += 1

    def Default_Model_Generator(self):
        # 缺省model
        model = tf.keras.Sequential(name='Default_Model')

        model.add(tf.keras.layers.Conv2D(
            input_shape=(self.inputFormat[1], self.inputFormat[2], self.inputFormat[3]),
            kernel_size=(8, 8), padding='SAME', activation='relu', filters=16, strides=(1, 1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)))

        model.add(tf.keras.layers.MaxPool2D(padding='SAME', pool_size=(2, 2), strides=(2, 2)))

        model.add(
            tf.keras.layers.Conv2D(kernel_size=(5, 5), padding='SAME', activation='relu', filters=32, strides=(1, 1),
                                   bias_initializer=tf.keras.initializers.Constant(0.1),
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)))

        model.add(tf.keras.layers.MaxPool2D(padding='SAME', pool_size=(1, 1), strides=(1, 1)))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(units=512, activation='relu',
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                        bias_initializer=tf.keras.initializers.Constant(0.1)))

        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(units=self.trainTargetNumber, activation='softmax',
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                        bias_initializer=tf.keras.initializers.Constant(0.1)))
        model.summary()

        return model

    def Model_Generator(self):
        pass

    def TrainModel(self, model):
        #
        time_begin = time.time()

        # 数据准备结束
        self.LoadData()
        time_elapsed = time.time() - time_begin
        print("读取test和valid %d图片文件耗费时间：%d秒" % (self.train_count + self.val_count, time_elapsed))

        # FIXME:
        if os.path.exists(self.logDir): shutil.rmtree(self.logDir)
        writer = tf.summary.create_file_writer(self.logDir)

        # 训练开始
        model.compile(optimizer=tf.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        history = model.fit(self.train_images, self.train_labels, epochs=self.iterations, batch_size=self.batch_size,
                            verbose=self.verbose, validation_split=self.validation_split)
        loss, acc = model.evaluate(self.val_images, self.val_labels)

        model.save(self.modelSavePath.__str__())

        # result/history 可视化
        fig = plt.figure()
        sub_fig = fig.add_subplot(1, 1, 1)
        sub_fig.plot(history.history['accuracy'])
        sub_fig.plot(history.history['val_accuracy'])
        sub_fig.legend(['training', 'validation'], loc='upper left')
        plt.show()

        time_elapsed = time.time() - time_begin
        print("训练和验证共耗费时间：%d秒" % time_elapsed)


    def PredictImg(self, img, modelPath = '-'):
        # 利用model预测
        modelPath = self.modelSavePath.__str__() if modelPath == "-" else modelPath.__str__()

        imgInput = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgInput = cv2.resize(imgInput, (self.WIDTH_Column, self.HEIGHT_Row))
        imgInput = imgInput.reshape(self.inputFormat)

        model = tf.keras.models.load_model(modelPath)
        result = np.array(model.predict(np.array(imgInput, dtype=np.float))[0])

        id = result.argmax()
        confidence = result[id]
        predictResult = self.trainTarget[id]

        return [predictResult, confidence]

class ProvinceTrain(Train):
    # 省份训练器

    def __init__(self, trainTargetNumber = 31,
                 modelSaveDir = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'model', 'provinces'),
                 logDir = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'log', 'provinces'),
                 trainDatasetDir = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'dataset',
                                                                 'train_images', 'training-set', 'chinese-characters'),
                 validDatasetDir = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'dataset',
                                                                 'train_images', 'validation-set', 'chinese-characters'),
                 trainTarget = ("京", "闽", "粤", "苏", "沪", "浙", "川", "鄂", "甘", "赣", "贵", "桂", "黑", "吉", "冀",
                                "津", "晋", "辽", "鲁", "蒙", "宁", "青", "琼", "陕", "皖", "湘", "新", "渝", "豫", "云", "藏"),
                 WIDTH_Column = 32,
                 HEIGHT_Row = 40,
                 CHANNEL = 1,
                 verbose=1,
                 validation_split=0.05,
                 batch_size = 60,
                 iterations = 10,
                 modelName = "model.h5"):
        super().__init__(trainTargetNumber = trainTargetNumber,
                         modelSaveDir = modelSaveDir,
                         logDir = logDir,
                         trainDatasetDir = trainDatasetDir,
                         validDatasetDir = validDatasetDir,
                         trainTarget = trainTarget,
                         WIDTH_Column = WIDTH_Column,
                         HEIGHT_Row = HEIGHT_Row,
                         CHANNEL = CHANNEL,
                         verbose = verbose,
                         validation_split=validation_split,
                         batch_size = batch_size,
                         iterations = iterations,
                         modelName = modelName)

class LettersTrain(Train):
    # 字母训练器

    def __init__(self, trainTargetNumber = 24,
                 modelSaveDir = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'model', 'letters'),
                 logDir = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'log', 'letters'),
                 trainDatasetDir=globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'dataset',
                                                               'train_images', 'training-set', 'letters'),
                 validDatasetDir=globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'dataset',
                                                               'train_images', 'validation-set', 'letters'),
                 trainTarget = ("A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                                "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y","Z"),
                 WIDTH_Column = 32,
                 HEIGHT_Row = 40,
                 CHANNEL = 1,
                 verbose=1,
                 batch_size = 60,
                 iterations = 10,
                 validation_split = 0.05,
                 modelName = "model.h5"):
        super().__init__(trainTargetNumber = trainTargetNumber,
                         modelSaveDir = modelSaveDir,
                         logDir=logDir,
                         trainDatasetDir = trainDatasetDir,
                         validDatasetDir = validDatasetDir,
                         trainTarget = trainTarget,
                         WIDTH_Column = WIDTH_Column,
                         HEIGHT_Row = HEIGHT_Row,
                         CHANNEL = CHANNEL,
                         verbose=verbose,
                         batch_size = batch_size,
                         iterations = iterations,
                         validation_split = validation_split,
                         modelName = modelName)

class DigitsTrain(Train):
    # 字母与数字训练器

    def __init__(self, trainTargetNumber=34,
                 modelSaveDir= globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'model', 'digits'),
                 logDir = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'log', 'digits'),
                 trainDatasetDir=globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'dataset',
                                                               'train_images', 'training-set'),
                 validDatasetDir=globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'dataset',
                                                               'train_images', 'validation-set'),
                 trainTarget=("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
                      "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"),
                 WIDTH_Column=32,
                 HEIGHT_Row=40,
                 CHANNEL=1,
                 verbose=1,
                 batch_size=60,
                 iterations=10,
                 validation_split = 0.05,
                 modelName="model.h5"):
        super().__init__(trainTargetNumber=trainTargetNumber,
                         modelSaveDir=modelSaveDir,
                         logDir=logDir,
                         trainDatasetDir=trainDatasetDir,
                         validDatasetDir=validDatasetDir,
                         trainTarget=trainTarget,
                         WIDTH_Column=WIDTH_Column,
                         HEIGHT_Row=HEIGHT_Row,
                         CHANNEL=CHANNEL,
                         verbose=verbose,
                         batch_size=batch_size,
                         iterations=iterations,
                         validation_split = validation_split,
                         modelName=modelName)


