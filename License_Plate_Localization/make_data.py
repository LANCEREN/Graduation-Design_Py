import os
import shutil
import Global_Var
import xml.dom.minidom as mnd

def DealXMLFile(filePath):
    domTree = mnd.parse(filePath)
    rootNode = domTree.documentElement
    print(rootNode.nodeName)

    # 所有annotation
    print("****部分annotation信息****")
    # folder 元素
    folder = rootNode.getElementsByTagName("folder")[0]
    print(folder.nodeName, ":", folder.childNodes[0].nodeValue)
    # phone 元素
    filename = rootNode.getElementsByTagName("filename")[0]
    print(filename.nodeName, ":", filename.childNodes[0].data)
    # size 元素
    size = rootNode.getElementsByTagName("size")[0]
    print(size.nodeName, ":", size.childNodes[0].data)
    height = size.getElementsByTagName("height")[0]
    width = size.getElementsByTagName("width")[0]
    depth = size.getElementsByTagName("depth")[0]

    content = Global_Var.projectPath + "License_Plate_Localization/dataset/dataset_voc/" + folder.childNodes[0].data \
              + "/" + filename.childNodes[0].data + " "
    # object 元素
    objects = rootNode.getElementsByTagName("object")
    for object in objects:
        print("****object信息****")
        # name 元素
        name = object.getElementsByTagName('name')[0]
        print(name.nodeName, ":", name.childNodes[0].data)
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


def SelectFileFromCCPD():
    ccpdPath = "/Users/lanceren/Downloads/GD_Dataset/Raw_Data/2019/CCPD2019/"
    targetFolder = "/Users/lanceren/Desktop/CCPD_Picts"
    targetFolder_Normal = "/Users/lanceren/Desktop/CCPD_Picts/Normal"
    targetFolder_SpecialCar = "/Users/lanceren/Desktop/CCPD_Picts/SpecialCar"
    targetFolder_Weather = "/Users/lanceren/Desktop/CCPD_Picts/Weather"

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

                index_of_low = [i for i, j in enumerate(filename) if j in ["_"]]
                index_of_middle = [i for i, j in enumerate(filename) if j in ["-"]]
                index_of_and = [i for i, j in enumerate(filename) if j in ["&"]]
                index_of_point = [i for i, j in enumerate(filename) if j in ["."]]

                horizon = int(filename[index_of_middle[0] + 1: index_of_low[0]])
                vertical = int(filename[index_of_low[0] + 1: index_of_middle[1]])
                xmin = int(filename[index_of_middle[1] + 1: index_of_and[0]])
                ymin = int(filename[index_of_and[0] + 1: index_of_low[1]])
                xmax = int(filename[index_of_low[1] + 1: index_of_and[1]])
                ymax = int(filename[index_of_and[1] + 1: index_of_middle[2]])
                province = int(filename[index_of_middle[3] + 1: index_of_low[5]])
                light = int(filename[index_of_middle[4] + 1: index_of_middle[-1]])
                blur = int(filename[index_of_middle[-1] + 1: index_of_point[0]])

                if horizon <= 0 and vertical <= 0 and light >= 100 and blur >= 100:
                    standCount += 1
                    shutil.copy(fullFileName, targetFolder_Normal)
                if province >= 31:
                    specialCarCount += 1
                    shutil.copy(fullFileName, targetFolder_SpecialCar)
                if index_of_low.count == 12:
                    newPowerCarCount += 1
                    shutil.copy(fullFileName, targetFolder_SpecialCar)
                if rt == ccpdPath + "ccpd_weather":
                    if horizon <= 2 and vertical <= 2 and blur <= 15:
                        weatherCount += 1
                        shutil.copy(fullFileName, targetFolder_Weather)

        print("new power : ", newPowerCarCount)
        print("specialCar : ", specialCarCount)
        print("weather : ", weatherCount)
        print("standCount : ", standCount)
        print("totalCount : ", totalCount)


namePath = "./data/classes/gd_detect.names"
if not os.path.exists(namePath):
    with open(namePath, "a+", encoding='utf-8') as f:
        f.writelines("Plate")
        f.close()

namePath = "./data/dataset/gd_detect_train.txt"
if not os.path.exists(namePath):
    with open(namePath, "a+", encoding='utf-8') as f:
        labelData = []
        labelDir = "./data/dataset/dataset_voc/Annotations/xml/"
        labelNum = 0
        for rt, dirs, files in os.walk(labelDir):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            for filename in files:
                Fullfilename = labelDir + filename
                content = DealXMLFile(Fullfilename) + os.linesep
                labelData.append(content)
                labelNum += 1
        f.writelines(labelData)
        f.close()

namePath = "./data/dataset/gd_detect_test.txt"
if not os.path.exists(namePath):
    with open(namePath, "a+", encoding='utf-8') as f:
        labelData = []
        labelDir = "./data/dataset/dataset_voc/Annotations/xml/"
        labelNum = 0
        for rt, dirs, files in os.walk(labelDir):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            for filename in files:
                Fullfilename = labelDir + filename
                content = DealXMLFile(Fullfilename) + os.linesep
                if (labelNum % 10) == 0:
                    labelData.append(content)
                labelNum += 1
        f.writelines(labelData)
        f.close()



