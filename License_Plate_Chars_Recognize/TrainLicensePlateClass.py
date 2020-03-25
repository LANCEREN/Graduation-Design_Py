import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import PIL
import Global_Var

class Train():
    # 训练器的父类

    def __init__(self, trainTargetNumber, Saver_Dir, Train_Dir, Val_Dir, TrainTarget,
                 WIDTH_Column= 32,
                 HEIGHT_Row = 40,
                 CHANNEL = 1,
                 batch_size = 60,
                 iterations = 10,
                 validation_split = 0.1,
                 verbose = 1,
                 Model_Name = "model.h5"
                 ):

        # 加载数据 进行训练的准备工作
        time_begin = time.time()

        self.WIDTH_Column = WIDTH_Column
        self.HEIGHT_Row = HEIGHT_Row
        self.SIZE = HEIGHT_Row * WIDTH_Column
        self.CHANNEL = CHANNEL
        self.InputFormat = [-1, HEIGHT_Row, WIDTH_Column, CHANNEL]
        self.TrainTarget = TrainTarget
        self.trainTargetNumber = trainTargetNumber

        self.batch_size = batch_size
        self.iterations = iterations
        self.verbose = verbose
        self.validation_split = validation_split
        self.Model_Name = Model_Name
        self.Saver_Dir = Saver_Dir

        self.Train_Dir = Train_Dir
        self.Val_Dir =  Val_Dir
        self.train_count = 0
        self.val_count = 0

        self.LoadData()
        # 数据准备结束
        time_elapsed = time.time() - time_begin
        print("读取test和valid %d图片文件耗费时间：%d秒" % (self.train_count + self.val_count, time_elapsed))

        #生成缺省model
        self.Default_Model_Generator()

    def LoadData(self):
        # 获取训练和评估图片总数
        for i in range(0, self.trainTargetNumber):
            dir = self.Train_Dir  + f'{i}/'
            for rt, dirs, files in os.walk(dir):
                for filename in files:
                    self.train_count += 1
        for i in range(0, self.trainTargetNumber):
            dir = self.Val_Dir + f'{i}/'
            for rt, dirs, files in os.walk(dir):
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
            dir = self.Train_Dir  + f'{i}/'
            for rt, dirs, files in os.walk(dir):
                for filename in files:
                    fullFileName = dir + filename
                    img = PIL.Image.open(fullFileName)
                    self.train_images[index] = np.array(img).reshape(self.InputFormat)
                    self.train_labels[index][i] = 1
                    # WIDTH_Column = img.size[0]
                    # HEIGHT_Row = img.size[1]
                    # for h in range(0, HEIGHT_Row):
                    #     for w in range(0, WIDTH_Column):
                    #         # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    #         if img.getpixel((w, h)) > 230:
                    #             self.train_images[index][w + h * WIDTH_Column] = 0
                    #         else:
                    #             self.train_images[index][w + h * WIDTH_Column] = img.getpixel((w, h))
                    index += 1

        # 生成评估测试的图片数据和标签
        index = 0
        for i in range(0, self.trainTargetNumber):
            dir = self.Val_Dir + f'{i}/'
            for rt, dirs, files in os.walk(dir):
                for filename in files:
                    fullFileName = dir + filename
                    img = PIL.Image.open(fullFileName)
                    self.val_images[index] = np.array(img).reshape(self.InputFormat)
                    self.val_labels[index][i] = 1
                    # WIDTH_Column = img.size[0]
                    # HEIGHT_Row = img.size[1]
                    # for h in range(0, HEIGHT_Row):
                    #     for w in range(0, WIDTH_Column):
                    #         # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    #         if img.getpixel((w, h)) > 230:
                    #             self.val_images[index][w + h * WIDTH_Column] = 0
                    #         else:
                    #             self.val_images[index][w + h * WIDTH_Column] = 1
                    index += 1

    def Default_Model_Generator(self):
        # 缺省model
        model = tf.keras.Sequential(name='Default_Model')

        model.add(tf.keras.layers.Conv2D(
            input_shape=(self.InputFormat[1], self.InputFormat[2], self.InputFormat[3]),
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

        model.compile(optimizer=tf.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        model.summary()

        self.Default_Model = model
        return model

    def Model_Generator(self):
        pass

    def TrainModel(self, model):
        # 训练开始
        time_begin = time.time()

        history = model.fit(self.train_images, self.train_labels, epochs=self.iterations, batch_size=self.batch_size,
                            verbose=self.verbose, validation_split=self.validation_split)
        loss, acc = model.evaluate(self.val_images, self.val_labels)

        model.save(self.Saver_Dir + self.Model_Name)

        # result/history 可视化
        fig = plt.figure()
        sub_fig = fig.add_subplot(1, 1, 1)
        sub_fig.plot(history.history['accuracy'])
        sub_fig.plot(history.history['val_accuracy'])
        sub_fig.legend(['training', 'validation'], loc='upper left')
        plt.show()

        time_elapsed = time.time() - time_begin
        print("训练和验证共耗费时间：%d秒" % time_elapsed)

    def PredictImg(self, ImgPath):
        # 利用model预测

        PredictResult = "-"
        img = PIL.Image.open(ImgPath)
        ImgInput = np.array(img).reshape(self.InputFormat)
        model = tf.keras.models.load_model(self.Saver_Dir + self.Model_Name)
        result = model.predict(x = ImgInput)
        for i in range(0,self.trainTargetNumber):
            if result[0][i] == 1.0:
                PredictResult = self.TrainTarget[i]
                break
        return PredictResult

class ProvinceTrain(Train):
    # 省份训练器

    def __init__(self, trainTargetNumber = 31,
                 Saver_Dir = Global_Var.projectPath + "/License_Plate_Chars_Recognize/train-saver/province/",
                 Train_Dir = Global_Var.projectPath + "/License_Plate_Chars_Recognize/LPCR_DataSet/train_images/training-set/chinese-characters/",
                 Val_Dir = Global_Var.projectPath + "/License_Plate_Chars_Recognize/LPCR_DataSet/train_images/validation-set/chinese-characters/",
                 TrainTarget = ("京", "闽", "粤", "苏", "沪", "浙", "川", "鄂", "甘", "赣", "贵", "桂", "黑", "吉", "冀",
                                "津", "晋", "辽", "鲁", "蒙", "宁", "青", "琼", "陕", "皖", "湘", "新", "渝", "豫", "云", "藏"),
                 WIDTH_Column = 32,
                 HEIGHT_Row = 40,
                 CHANNEL = 1,
                 verbose=1,
                 validation_split=0.05,
                 batch_size = 60,
                 iterations = 10,
                 Model_Name = "model.h5"):
        super().__init__(trainTargetNumber = trainTargetNumber,
                         Saver_Dir = Saver_Dir,
                         Train_Dir = Train_Dir,
                         Val_Dir = Val_Dir,
                         TrainTarget = TrainTarget,
                         WIDTH_Column = WIDTH_Column,
                         HEIGHT_Row = HEIGHT_Row,
                         CHANNEL = CHANNEL,
                         validation_split=validation_split,
                         batch_size = batch_size,
                         iterations = iterations,
                         Model_Name = Model_Name)

class LettersTrain(Train):
    # 字母训练器

    def __init__(self, trainTargetNumber = 24,
                 Saver_Dir = Global_Var.projectPath + "/License_Plate_Chars_Recognize/train-saver/letters/",
                 Train_Dir = Global_Var.projectPath + "/License_Plate_Chars_Recognize/LPCR_DataSet/train_images/training-set/letters/",
                 Val_Dir = Global_Var.projectPath + "/License_Plate_Chars_Recognize/LPCR_DataSet/train_images/validation-set/letters/",
                 TrainTarget = ("A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                                "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y","Z"),
                 WIDTH_Column = 32,
                 HEIGHT_Row = 40,
                 CHANNEL = 1,
                 verbose=1,
                 batch_size = 60,
                 iterations = 10,
                 validation_split = 0.05,
                 Model_Name = "model.h5"):
        super().__init__(trainTargetNumber = trainTargetNumber,
                         Saver_Dir = Saver_Dir,
                         Train_Dir = Train_Dir,
                         Val_Dir = Val_Dir,
                         TrainTarget = TrainTarget,
                         WIDTH_Column = WIDTH_Column,
                         HEIGHT_Row = HEIGHT_Row,
                         CHANNEL = CHANNEL,
                         batch_size = batch_size,
                         iterations = iterations,
                         validation_split = validation_split,
                         Model_Name = Model_Name)

class DigitsTrain(Train):
    # 字母与数字训练器

    def __init__(self, trainTargetNumber=34,
                 Saver_Dir= Global_Var.projectPath + "/License_Plate_Chars_Recognize/train-saver/digits/",
                 Train_Dir= Global_Var.projectPath + "/License_Plate_Chars_Recognize/"+
                           "LPCR_DataSet/train_images/training-set/",
                 Val_Dir= Global_Var.projectPath + "/License_Plate_Chars_Recognize/LPCR_DataSet/train_images/validation-set/",
                 TrainTarget=("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
                      "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"),
                 WIDTH_Column=32,
                 HEIGHT_Row=40,
                 CHANNEL=1,
                 verbose=1,
                 batch_size=60,
                 iterations=10,
                 validation_split = 0.05,
                 Model_Name="model.h5"):
        super().__init__(trainTargetNumber=trainTargetNumber,
                         Saver_Dir=Saver_Dir,
                         Train_Dir=Train_Dir,
                         Val_Dir=Val_Dir,
                         TrainTarget=TrainTarget,
                         WIDTH_Column=WIDTH_Column,
                         HEIGHT_Row=HEIGHT_Row,
                         CHANNEL=CHANNEL,
                         batch_size=batch_size,
                         iterations=iterations,
                         validation_split = validation_split,
                         Model_Name=Model_Name)


