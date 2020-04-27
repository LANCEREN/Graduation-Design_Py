import os,cv2
import numpy as np
from pathlib2 import Path
from License_Plate_Color_Recognize import color_train
from License_Plate_Color_Recognize.core.config import cfg


def Lpcor_Operator(img, fileName):
    """进行车牌颜色识别License_Plate_Color_Recognize"""

    trainClass = color_train.Train()
    predictResult = trainClass.SimplePredict(img)
    return trainClass.plateType[predictResult]



if __name__ == "__main__":
    # dirPath = r"/Users/lanceren/Desktop/has"
    # for rt, dirs, files in os.walk(dirPath):
    #     files = [f for f in files if not f[0] == '.']
    #     dirs[:] = [d for d in dirs if not d[0] == '.']
    #     for file in files:
    #         filePath = Path(rt, file)
    #         predictResult = trainClass.SimplePredict(filePath, model)
    #         print(predictResult)
    #         cv2.imwrite(f"/Users/lanceren/PycharmProjects/LPR_OpenCV_Graduation/License_Plate_Color_Recognize/data/dataset/{predictResult}/{fileName}.jpg",img)
    pass