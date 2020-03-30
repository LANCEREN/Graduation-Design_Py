import os, sys
import global_var
import PIL
from PIL import Image


class PrepareData():

    def __init__(self):

        self.saverDirTrain = global_var.projectPath + "/" + "License_Plate_Chars_Recognize/" + "LPCR_DataSet/train_images/training-set/"
        self.saverDirVal = global_var.projectPath + "/" + "License_Plate_Chars_Recognize/" + "LPCR_DataSet/train_images/validation-set/"
        self.inputDirDigit = ["/Users/lanceren/Downloads/毕设实验数据/LPCR数据/训练字符识别模型数据/",
                              "/Users/lanceren/Downloads/毕设实验数据/tf_car_license_dataset/train_images/validation-set/"
            , "/Users/lanceren/Downloads/毕设实验数据/LPCR数据/car_charactes少/digit_letters/",
                              "/Users/lanceren/Downloads/毕设实验数据/tf_car_license_dataset/train_images/training-set/"]
        self.inputDirLetter = [
            "/Users/lanceren/Downloads/毕设实验数据/tf_car_license_dataset/train_images/training-set/letters/"]
        self.inputDirProvince = ["/Users/lanceren/Downloads/毕设实验数据/LPCR数据/训练字符识别模型数据/中文/",
                                 "/Users/lanceren/Downloads/毕设实验数据/LPCR数据/car_charactes少/chinese/"]
        self.trainTargetNumber = 34

    def InputToTrain(self):

        # 生成Digit训练图片数据和标签
        for i in self.inputDirDigit:
            index = 0
            for j in range(0, self.trainTargetNumber):
                InputDir = i + f'{j}/'
                SaveDir = self.saverDirTrain + f"{j}/"
                if not os.path.exists(SaveDir):
                    os.mkdir(SaveDir)
                for rt, dirs, files in os.walk(InputDir):
                    for filename in files:
                        fullFileName = InputDir + filename
                        img = PIL.Image.open(fullFileName)
                        newimg = img.resize((32, 40), PIL.Image.ANTIALIAS)
                        newimg = newimg.convert('L')
                        newimg.save(SaveDir + f"{filename}.png")
                        index += 1

        # 生成Letter训练图片数据和标签
        for i in self.inputDirLetter:
            index = 0
            for j in range(10, self.trainTargetNumber):
                InputDir = i + f'{j}/'
                SaveDir = self.saverDirTrain + "letters/" + f"{j - 10}/"
                if not os.path.exists(SaveDir):
                    os.mkdir(SaveDir)
                for rt, dirs, files in os.walk(InputDir):
                    for filename in files:
                        fullFileName = InputDir + filename
                        img = PIL.Image.open(fullFileName)
                        newimg = img.resize((32, 40), PIL.Image.ANTIALIAS)
                        newimg = newimg.convert('L')
                        newimg.save(self.saverDirTrain + f"{j}/" + f"{filename}.png")
                        index += 1

        for i in range(0, 24):
            index = 0
            InputDir = self.saverDirTrain + f'{i}/'
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    if index % 3 == 0:
                        fullFileName = InputDir + filename
                        img = PIL.Image.open(fullFileName)
                        img.save(self.saverDirTrain + "letters/" + f"{i}/" + f"{filename}.png")
                    index += 1

        # 生成Province训练图片数据和标签
        for i in self.inputDirProvince:
            index = 0
            for j in range(0, 31):
                InputDir = i + f'{j}/'
                SaveDir = self.saverDirTrain + "chinese-characters/" + f"{j}/"
                if not os.path.exists(SaveDir):
                    os.mkdir(SaveDir)
                for rt, dirs, files in os.walk(InputDir):
                    for filename in files:
                        fullFileName = InputDir + filename
                        img = PIL.Image.open(fullFileName)
                        newimg = img.resize((32, 40), PIL.Image.ANTIALIAS)
                        newimg = newimg.convert('L')
                        newimg.save(SaveDir + f"{filename}.png")
                        index += 1

        index = 0
        for j in range(0, 6):
            InputDir = "/Users/lanceren/Downloads/毕设实验数据/tf_car_license_dataset/train_images/training-set/chinese-characters" + f'{j}/'
            SaveDir = self.saverDirTrain + "chinese-characters/" + f"{j}/"
            if not os.path.exists(SaveDir):
                os.mkdir(SaveDir)
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    fullFileName = InputDir + filename
                    img = PIL.Image.open(fullFileName)
                    newimg = img.resize((32, 40), PIL.Image.ANTIALIAS)
                    newimg = newimg.convert('L')
                    newimg.save(SaveDir + f"{filename}.png")
                    index += 1

    def SampleToValid(self):

        # 生成Digit评估测试的图片数据和标签

        for i in range(0, self.trainTargetNumber):
            index = 0
            InputDir = self.saverDirTrain + f'{i}/'
            if not os.path.exists(self.saverDirVal + f"{i}/"):
                os.mkdir(self.saverDirVal + f"{i}/")
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    if index % 9 == 0:
                        fullFileName = InputDir + filename
                        img = PIL.Image.open(fullFileName)
                        img.save(self.saverDirVal + f"{i}/" + f"{filename}.png")
                    index += 1

        # 生成Letters评估测试的图片数据和标签

        for i in range(0, 24):
            index = 0
            InputDir = self.saverDirTrain + "letters/" + f'{i}/'
            if not os.path.exists(self.saverDirVal + "letters/" + f"{i}/"):
                os.mkdir(self.saverDirVal + "letters/" + f"{i}/")
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    if index % 9 == 0:
                        fullFileName = InputDir + filename
                        img = PIL.Image.open(fullFileName)
                        img.save(self.saverDirVal + "letters/" + f"{i}/" + f"{filename}.png")
                    index += 1

        for i in range(0, 31):
            index = 0
            InputDir = self.saverDirTrain + "chinese-characters/" + f'{i}/'
            if not os.path.exists(self.saverDirVal + "chinese-characters/" + f"{i}/"):
                os.mkdir(self.saverDirVal + "chinese-characters/" + f"{i}/")
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    if index % 9 == 0:
                        fullFileName = InputDir + filename
                        img = PIL.Image.open(fullFileName)
                        img.save(self.saverDirVal + "chinese-characters/" + f"{i}/" + f"{filename}.png")
                    index += 1


if __name__ == "__main__":
    a = PrepareData()
    a.InputToTrain()
    a.SampleToValid()
