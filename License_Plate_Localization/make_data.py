import os, platform
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
    content = (globalVars.projectPath / Path('License_Plate_Localization', 'data', 'dataset', folder.childNodes[0].data,
                                             filename.childNodes[0].data)).__str__() + " "

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
    ccpdPath = Path("/Users/lanceren/Downloads/GD_Dataset/Raw_Data/2019/CCPD2019")
    targetFolder = Path("/Users/lanceren/Desktop/CCPD_Picts/")
    targetFolder_Normal = targetFolder / Path('Normal')
    targetFolder_SpecialCar = targetFolder / Path('SpecialCar')
    targetFolder_Weather = targetFolder / Path('Weather')

    if os.path.exists(targetFolder.__str__()): shutil.rmtree(targetFolder.__str__())
    os.mkdir(targetFolder.__str__())
    os.mkdir(targetFolder_Normal.__str__())
    os.mkdir(targetFolder_SpecialCar.__str__())
    os.mkdir(targetFolder_Weather.__str__())

    if os.path.exists(ccpdPath.__str__()):
        totalCount = 0
        standCount = 0
        specialCarCount = 0
        newPowerCarCount = 0
        weatherCount = 0
        for rt, dirs, files in os.walk(ccpdPath.__str__()):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            if rt == (ccpdPath / Path('ccpd_np')).__str__():
                continue
            if rt == (ccpdPath / Path('ccpd_base')).__str__():
                for filename in files:
                    totalCount += 1
                    fullFileName = Path(rt, filename)
                    ccpdName = CCPDNameParams(filename)

                    if ccpdName.horizon <= 10 and ccpdName.vertical <= 10 and ccpdName.light >= 100 and ccpdName.blur >= 100:
                        standCount += 1
                        shutil.copy(fullFileName.__str__(), targetFolder_Normal)
                    if ccpdName.province >= 31:
                        specialCarCount += 1
                        shutil.copy(fullFileName.__str__(), targetFolder_SpecialCar)
                    if ccpdName.index_of_low.count == 12:
                        newPowerCarCount += 1
                        shutil.copy(fullFileName.__str__(), targetFolder_SpecialCar)
            if rt == (ccpdPath / Path('ccpd_weather')).__str__():
                continue
                if ccpdName.horizon <= 2 and ccpdName.vertical <= 2 and ccpdName.blur <= 15:
                    weatherCount += 1
                    shutil.copy(fullFileName.__str__(), targetFolder_Weather)

        print("new power : ", newPowerCarCount)
        print("specialCar : ", specialCarCount)
        print("weather : ", weatherCount)
        print("standCount : ", standCount)
        print("totalCount : ", totalCount)


def CreateDotNames():
    # create gd_detect.names
    dotNamePath = globalVars.projectPath / Path('License_Plate_Localization', 'data', 'classes', 'gd_detect.names')
    if os.path.exists(dotNamePath.__str__()): shutil.rmtree(dotNamePath.__str__())
    os.mkdir(dotNamePath.__str__())
    with open(dotNamePath.__str__(), "w+", encoding='utf-8') as f:
        f.writelines("Plate")
        f.close()


def CreateLabelTxt():
    def generateLabelTxtInMode(labelTxtPath_key, labelTxtPath_value, annotationDirPathDict):
        def generateLabelTxtBySource(mode, annotationDirPath_key, annotationDirPath_value, data):
            def generateContent(source, fullFilePath):
                def case1():
                    return DealXMLFile(fullFilePath.__str__())

                def case2():
                    ccpdName = CCPDNameParams(fullFilePath.name)
                    return fullFilePath.__str__() + " " + ccpdName.CCPDNameToLabelProcess()

                def default():
                    print("mode error!")

                switch = {'labelMe': case1,
                          'ccpd': case2}
                choice = source  # 获取选择
                content = switch.get(choice, default)() + os.linesep  # 执行对应的函数，如果没有就执行默认的函数
                return content

            labelNum = 0
            modeNum = 1 if mode == "train" else 10  # train全取，test十取一
            for rt, dirs, files in os.walk(annotationDirPath_value.__str__()):
                files = [f for f in files if not f[0] == '.']
                dirs[:] = [d for d in dirs if not d[0] == '.']
                for filename in files:
                    fullFileName = Path(rt, filename)
                    content = generateContent(annotationDirPath_key, fullFileName)
                    if labelNum % modeNum == 0: data.append(content)
                    labelNum += 1

        labelData = []
        for key, value in annotationDirPathDict.items():
            generateLabelTxtBySource(labelTxtPath_key, key, value, labelData)
        if os.path.exists(labelTxtPath_value.__str__()): os.remove(labelTxtPath_value.__str__())
        with open(labelTxtPath_value.__str__(), "w+", encoding='utf-8') as f:
            f.writelines(labelData)
            f.close()

    operatingSystem = 'dos' if platform.system() == 'Windows' else 'unix'  # 获取选择
    annotationDirPath = {
        "labelMe": globalVars.projectPath / Path('License_Plate_Localization', 'data', 'annotations', 'xml'),
        "ccpd": globalVars.projectPath / Path('License_Plate_Localization', 'data', 'dataset', 'JPEGImages',
                                              'ccpd_sample')}
    labelTxtPath = {
        "train": globalVars.projectPath / Path('License_Plate_Localization', 'data', 'labels', f'{operatingSystem}',
                                               'gd_detect_train.txt'),
        "test": globalVars.projectPath / Path('License_Plate_Localization', 'data', 'labels', f'{operatingSystem}',
                                              'gd_detect_test.txt')}

    for key, value in labelTxtPath.items():
        generateLabelTxtInMode(key, value, annotationDirPath)

if __name__ == "__main__":
    # SelectFileFromCCPD()
    # CreateDotNames()
    CreateLabelTxt()
