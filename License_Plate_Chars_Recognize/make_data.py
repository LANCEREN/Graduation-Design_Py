import os, sys
import shutil
from pathlib2 import Path
from global_var import globalVars
import PIL
from PIL import Image


class PrepareData():

    def __init__(self):

        self.saverDirTrain = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'dataset',
                                                           'train_images', 'training-set')
        self.saverDirVal = globalVars.projectPath / Path('License_Plate_Chars_Recognize', 'data', 'dataset',
                                                         'train_images', 'validation-set')

    def SampleToValid(self, mode='digits'):

        # 判断mode
        def case1():  # 第一种情况执行的函数
            self.inputDir = self.saverDirTrain
            self.saveDir = self.saverDirVal
            self.trainTargetNumber = 34

        def case2():  # 第二种情况执行的函数
            self.inputDir = self.saverDirTrain / Path('letters')
            self.saveDir = self.saverDirVal / Path('letters')
            self.trainTargetNumber = 24

        def case3():  # 第三种情况执行的函数
            self.inputDir = self.saverDirTrain / Path('chinese-characters')
            self.saveDir = self.saverDirVal / Path('chinese-characters')
            self.trainTargetNumber = 31

        def default():  # 默认情况下执行的函数
            print('No such case')

        switch = {'digits': case1,  # 注意此处不要加括号
                  'letters': case2,  # 注意此处不要加括号
                  'provinces': case3,  # 注意此处不要加括号
                  }

        choice = mode  # 获取选择
        switch.get(choice, default)()  # 执行对应的函数，如果没有就执行默认的函数

        # 生成 mode 评估测试的图片数据和标签
        for i in range(0, self.trainTargetNumber):
            index = 0
            inputDir = self.inputDir / Path(f'{i}')
            saveDir = self.saveDir / Path(f'{i}')
            if os.path.exists(saveDir.__str__()): shutil.rmtree(saveDir.__str__())
            os.mkdir(saveDir.__str__())
            for rt, dirs, files in os.walk(inputDir.__str__()):
                for filename in files:
                    if index % 9 == 0:
                        fullFileName = inputDir / Path(filename)
                        saveFile = saveDir / Path(f'{filename}')
                        img = PIL.Image.open(fullFileName.__str__())
                        img.save(saveFile.__str__())
                    index += 1


if __name__ == "__main__":
    a = PrepareData()
    a.SampleToValid()
