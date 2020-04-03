import os, sys
import shutil
from pathlib2 import Path
from global_var import globalVars
import xml.dom.minidom as mnd


def DealXMLFile(filePath):
    domTree = mnd.parse(filePath)
    # 所有annotation
    rootNode = domTree.documentElement
    # folder 元素
    folder = rootNode.getElementsByTagName("folder")[0]
    # phone 元素
    filename = rootNode.getElementsByTagName("filename")[0]
    # size 元素
    size = rootNode.getElementsByTagName("size")[0]
    height = size.getElementsByTagName("height")[0]
    width = size.getElementsByTagName("width")[0]
    depth = size.getElementsByTagName("depth")[0]
    content = folder.childNodes[0].data + "/" + filename.childNodes[0].data + " "

    # object 元素
    objects = rootNode.getElementsByTagName("object")
    for object in objects:
        # name 元素
        name = object.getElementsByTagName('name')[0]
        # bndbox 元素
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = bndbox.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = bndbox.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = bndbox.getElementsByTagName("ymax")[0].childNodes[0].data
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)
        content += f"{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)},0 "
    return content


class CCPDNameParams():
    def __init__(self, filename):
        self.index_of_low = [i for i, j in enumerate(filename) if j in ["_"]]
        self.index_of_middle = [i for i, j in enumerate(filename) if j in ["-"]]
        self.index_of_and = [i for i, j in enumerate(filename) if j in ["&"]]
        self.index_of_point = [i for i, j in enumerate(filename) if j in ["."]]

        self.horizon = int(filename[self.index_of_middle[0] + 1: self.index_of_low[0]])
        self.vertical = int(filename[self.index_of_low[0] + 1: self.index_of_middle[1]])
        self.xmin = int(filename[self.index_of_middle[1] + 1: self.index_of_and[0]])
        self.ymin = int(filename[self.index_of_and[0] + 1: self.index_of_low[1]])
        self.xmax = int(filename[self.index_of_low[1] + 1: self.index_of_and[1]])
        self.ymax = int(filename[self.index_of_and[1] + 1: self.index_of_middle[2]])
        self.province = int(filename[self.index_of_middle[3] + 1: self.index_of_low[5]])
        self.light = int(filename[self.index_of_middle[4] + 1: self.index_of_middle[-1]])
        self.blur = int(filename[self.index_of_middle[-1] + 1: self.index_of_point[0]])

    def CCPDNameToLabelProcess(self):
        content = f"{self.xmin},{self.ymin},{self.xmax},{self.ymax},0 "
        return content


def SelectFileFromCCPD():
    ccpdPath = "/Users/lanceren/Downloads/GD_Dataset/Raw_Data/2019/CCPD2019/"
    targetFolder = "/Users/lanceren/Desktop/CCPD_Picts/"
    targetFolder_Normal = "/Users/lanceren/Desktop/CCPD_Picts/Normal/"
    targetFolder_SpecialCar = "/Users/lanceren/Desktop/CCPD_Picts/SpecialCar/"
    targetFolder_Weather = "/Users/lanceren/Desktop/CCPD_Picts/Weather/"

    if os.path.exists(targetFolder): shutil.rmtree(targetFolder)
    os.mkdir(targetFolder)
    os.mkdir(targetFolder_Normal)
    os.mkdir(targetFolder_SpecialCar)
    os.mkdir(targetFolder_Weather)

    if os.path.exists(ccpdPath):
        totalCount = 0
        standCount = 0
        specialCarCount = 0
        newPowerCarCount = 0
        weatherCount = 0
        for rt, dirs, files in os.walk(ccpdPath):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            if rt == ccpdPath + "ccpd_np":
                continue

            for filename in files:
                totalCount += 1
                fullFileName = rt + "/" + filename
                ccpdName = CCPDNameParams(filename)

                if ccpdName.horizon <= 0 and ccpdName.vertical <= 0 and ccpdName.light >= 100 and ccpdName.blur >= 100:
                    standCount += 1
                    shutil.copy(fullFileName, targetFolder_Normal)
                if ccpdName.province >= 31:
                    specialCarCount += 1
                    shutil.copy(fullFileName, targetFolder_SpecialCar)
                if ccpdName.index_of_low.count == 12:
                    newPowerCarCount += 1
                    shutil.copy(fullFileName, targetFolder_SpecialCar)
                if rt == ccpdPath + "ccpd_weather":
                    if ccpdName.horizon <= 2 and ccpdName.vertical <= 2 and ccpdName.blur <= 15:
                        weatherCount += 1
                        shutil.copy(fullFileName, targetFolder_Weather)

        print("new power : ", newPowerCarCount)
        print("specialCar : ", specialCarCount)
        print("weather : ", weatherCount)
        print("standCount : ", standCount)
        print("totalCount : ", totalCount)


def CreateDotNames():
    dotNamePath = global_var.projectPath + "/" + "License_Plate_Localization/" + "data/classes/gd_detect.names"
    if not os.path.exists(dotNamePath):
        with open(dotNamePath, "w+", encoding='utf-8') as f:
            f.writelines("Plate")
            f.close()


def CreateLabelTxt(mode="train"):
    LabelTxtPath = global_var.projectPath + "/" + "License_Plate_Localization/" + "data/dataset/gd_detect_train.txt" if mode == "train" else \
        global_var.projectPath + "/" + "License_Plate_Localization/" + "data/dataset/gd_detect_test.txt"
    if os.path.exists(LabelTxtPath):os.remove(LabelTxtPath)
    with open(LabelTxtPath, "w+", encoding='utf-8') as f:
        labelData = []
        labelNum = 0

        labelDir = global_var.projectPath + "/" + "License_Plate_Localization/" + "data/dataset/dataset_voc/Annotations/xml/"
        for rt, dirs, files in os.walk(labelDir):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            for filename in files:
                fullFileName = rt + filename
                content = global_var.projectPath + "/" + "License_Plate_Localization/data/dataset/" + DealXMLFile(fullFileName) + os.linesep
                if mode == "train":
                    labelData.append(content)
                else:
                    if labelNum % 10 == 0:
                        labelData.append(content)
                labelNum += 1

        labelDir = global_var.projectPath + "/" + "License_Plate_Localization/" + "data/dataset/JPEGImages/ccpd_Normal/"
        for rt, dirs, files in os.walk(labelDir):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            for filename in files:
                fullFileName = rt + filename
                ccpdName = CCPDNameParams(filename)
                content = fullFileName + " " + ccpdName.CCPDNameToLabelProcess() + os.linesep
                if mode == "train":
                    labelData.append(content)
                else:
                    if labelNum % 10 == 0:
                        labelData.append(content)
                labelNum += 1

        f.writelines(labelData)
        f.close()


if __name__ == "__main__":
    CreateDotNames()
    CreateLabelTxt("train")
    CreateLabelTxt("test")
