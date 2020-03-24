import os
import cv2
import numpy as np
import shutil
import random
import argparse
import Global_Var
import xml.dom.minidom as dom

def DealXMLFile(filePath):
    domTree = dom.parse(filePath)
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

    content = Global_Var.projectPath + "License_Plate_Localization/dataset/dataset_voc/JPEGImages/" + filename.childNodes[0].data + " "
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



