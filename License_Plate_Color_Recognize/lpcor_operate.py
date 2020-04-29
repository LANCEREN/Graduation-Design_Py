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
    pass