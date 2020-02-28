import PIL
from PIL import Image
import os
import Global_Var

class PrepareData():

    def __init__(self):

        self.Saver_Dir_train = Global_Var.Project_Path + "/License_Plate_Chars_Recognize/" + "LPCR_DataSet/train_images/training-set/"
        self.Saver_Dir_Val = Global_Var.Project_Path + "/License_Plate_Chars_Recognize/LPCR_DataSet/train_images/validation-set/"
        self.Input_Dir_Digit = ["/Users/lanceren/Downloads/毕设实验数据/LPCR数据/训练字符识别模型数据/","/Users/lanceren/Downloads/毕设实验数据/tf_car_license_dataset/train_images/validation-set/"
            ,"/Users/lanceren/Downloads/毕设实验数据/LPCR数据/car_charactes少/digit_letters/","/Users/lanceren/Downloads/毕设实验数据/tf_car_license_dataset/train_images/training-set/"]
        self.Input_Dir_Letter = ["/Users/lanceren/Downloads/毕设实验数据/tf_car_license_dataset/train_images/training-set/letters/"]
        self.Input_Dir_Province = ["/Users/lanceren/Downloads/毕设实验数据/LPCR数据/训练字符识别模型数据/中文/","/Users/lanceren/Downloads/毕设实验数据/LPCR数据/car_charactes少/chinese/"]
        self.TrainTargetNumber = 34

    def inputToTrain(self):

        # 生成Digit训练图片数据和标签
        for i in self.Input_Dir_Digit:
            index = 0
            for j in range(0, self.TrainTargetNumber):
                InputDir = i + f'{j}/'
                SaveDir = self.Saver_Dir_train + f"{j}/"
                if os.path.exists(SaveDir) == False:
                    os.mkdir(SaveDir)
                for rt, dirs, files in os.walk(InputDir):
                    for filename in files:
                        Fullfilename = InputDir + filename
                        img = PIL.Image.open(Fullfilename)
                        newimg = img.resize((32, 40), PIL.Image.ANTIALIAS)
                        newimg = newimg.convert('L')
                        newimg.save(SaveDir + f"{filename}.png")
                        index += 1

        # 生成Letter训练图片数据和标签
        for i in self.Input_Dir_Letter:
            index = 0
            for j in range(10, self.TrainTargetNumber):
                InputDir = i + f'{j}/'
                SaveDir = self.Saver_Dir_train + "letters/" + f"{j-10}/"
                if os.path.exists(SaveDir) == False:
                    os.mkdir(SaveDir)
                for rt, dirs, files in os.walk(InputDir):
                    for filename in files:
                        Fullfilename = InputDir + filename
                        img = PIL.Image.open(Fullfilename)
                        newimg = img.resize((32, 40), PIL.Image.ANTIALIAS)
                        newimg = newimg.convert('L')
                        newimg.save(self.Saver_Dir_train + f"{j}/" + f"{filename}.png")
                        index += 1

        for i in range(0, 24):
            index = 0
            InputDir = self.Saver_Dir_train + f'{i}/'
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    if index % 3 == 0:
                        Fullfilename = InputDir + filename
                        img = PIL.Image.open(Fullfilename)
                        img.save(self.Saver_Dir_train + "letters/" + f"{i}/" + f"{filename}.png")
                    index += 1


        # 生成Province训练图片数据和标签
        for i in self.Input_Dir_Province:
            index = 0
            for j in range(0, 31):
                InputDir = i + f'{j}/'
                SaveDir = self.Saver_Dir_train + "chinese-characters/" + f"{j}/"
                if os.path.exists(SaveDir) == False:
                    os.mkdir(SaveDir)
                for rt, dirs, files in os.walk(InputDir):
                    for filename in files:
                        Fullfilename = InputDir + filename
                        img = PIL.Image.open(Fullfilename)
                        newimg = img.resize((32, 40), PIL.Image.ANTIALIAS)
                        newimg = newimg.convert('L')
                        newimg.save(SaveDir + f"{filename}.png")
                        index += 1

        index = 0
        for j in range(0, 6):
            InputDir = "/Users/lanceren/Downloads/毕设实验数据/tf_car_license_dataset/train_images/training-set/chinese-characters" + f'{j}/'
            SaveDir = self.Saver_Dir_train + "chinese-characters/" + f"{j}/"
            if os.path.exists(SaveDir) == False:
                os.mkdir(SaveDir)
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    Fullfilename = InputDir + filename
                    img = PIL.Image.open(Fullfilename)
                    newimg = img.resize((32, 40), PIL.Image.ANTIALIAS)
                    newimg = newimg.convert('L')
                    newimg.save(SaveDir + f"{filename}.png")
                    index += 1




    def sampleToValid(self):

        # 生成Digit评估测试的图片数据和标签

        for i in range(0, self.TrainTargetNumber):
            index = 0
            InputDir = self.Saver_Dir_train + f'{i}/'
            if os.path.exists(self.Saver_Dir_Val + f"{i}/") == False:
                os.mkdir(self.Saver_Dir_Val + f"{i}/")
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    if index % 9 == 0:
                        Fullfilename = InputDir + filename
                        img = PIL.Image.open(Fullfilename)
                        img.save(self.Saver_Dir_Val + f"{i}/" + f"{filename}.png")
                    index += 1

        # 生成Letters评估测试的图片数据和标签

        for i in range(0, 24):
            index = 0
            InputDir = self.Saver_Dir_train + "letters/" + f'{i}/'
            if os.path.exists(self.Saver_Dir_Val + "letters/" + f"{i}/") == False:
                os.mkdir(self.Saver_Dir_Val + "letters/" + f"{i}/")
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    if index % 9 == 0:
                        Fullfilename = InputDir + filename
                        img = PIL.Image.open(Fullfilename)
                        img.save(self.Saver_Dir_Val + "letters/" + f"{i}/" + f"{filename}.png")
                    index += 1

        for i in range(0, 31):
            index = 0
            InputDir = self.Saver_Dir_train + "chinese-characters/" + f'{i}/'
            if os.path.exists(self.Saver_Dir_Val + "chinese-characters/" + f"{i}/") == False:
                os.mkdir(self.Saver_Dir_Val + "chinese-characters/" + f"{i}/")
            for rt, dirs, files in os.walk(InputDir):
                for filename in files:
                    if index % 9 == 0:
                        Fullfilename = InputDir + filename
                        img = PIL.Image.open(Fullfilename)
                        img.save(self.Saver_Dir_Val + "chinese-characters/" + f"{i}/" + f"{filename}.png")
                    index += 1

if __name__ == "__main__" :
    a = PrepareData()
    a.inputToTrain()
    a.sampleToValid()