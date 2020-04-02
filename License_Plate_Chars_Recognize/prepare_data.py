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
        self.trainTargetNumber = 34

    def SampleToValid(self):

        # 生成Digit评估测试的图片数据和标签

        for i in range(0, self.trainTargetNumber):
            index = 0
            inputDir = self.saverDirTrain / Path(f'{i}')
            saveDir = self.saverDirVal / Path(f'{i}')
            if os.path.exists(saveDir.__str__()): shutil.rmtree(saveDir.__str__())
            os.mkdir(saveDir.__str__())
            for rt, dirs, files in os.walk(inputDir.__str__()):
                for filename in files:
                    if index % 9 == 0:
                        fullFileName = inputDir / Path(filename)
                        saveFile = saveDir / Path(f'{filename}.png')
                        img = PIL.Image.open(fullFileName.__str__())
                        img.save(saveFile.__str__())
                    index += 1

        # 生成Letters评估测试的图片数据和标签

        for i in range(0, 24):
            index = 0
            inputDir = self.saverDirTrain / Path('letters', f'{i}')
            saveDir = self.saverDirVal / Path('letters', f'{i}')
            if os.path.exists(saveDir.__str__()): shutil.rmtree(saveDir.__str__())
            os.mkdir(saveDir.__str__())
            for rt, dirs, files in os.walk(inputDir.__str__()):
                for filename in files:
                    if index % 9 == 0:
                        fullFileName = inputDir / Path(filename)
                        saveFile = saveDir / Path(f'{filename}.png')
                        img = PIL.Image.open(fullFileName.__str__())
                        img.save(saveFile.__str__())
                    index += 1

        # 生成Provinces评估测试的图片数据和标签

        for i in range(0, 31):
            index = 0
            inputDir = self.saverDirTrain / Path('chinese-characters', f'{i}')
            saveDir = self.saverDirVal / Path('chinese-characters', f'{i}')
            if os.path.exists(saveDir.__str__()): shutil.rmtree(saveDir.__str__())
            os.mkdir(saveDir.__str__())
            for rt, dirs, files in os.walk(inputDir.__str__()):
                for filename in files:
                    if index % 9 == 0:
                        fullFileName = inputDir / Path(filename)
                        saveFile = saveDir / Path(f'{filename}.png')
                        img = PIL.Image.open(fullFileName.__str__())
                        img.save(saveFile.__str__())
                    index += 1


if __name__ == "__main__":
    a = PrepareData()
    a.SampleToValid()
