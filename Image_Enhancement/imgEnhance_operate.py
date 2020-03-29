import cv2
import Global_Var
from Image_Enhancement import DarkChannelPrior

def Ie_Operator(i):

    m = DarkChannelPrior.deHaze(cv2.imread(Global_Var.projectPath + f'/Image_Enhancement/IE_DataSet/{i}.jpg') / 255.0) * 255
    cv2.imwrite(Global_Var.projectPath + f'/Image_Enhancement/IE_DataSet/{i}_.png', m)


if __name__ == "__main__":
    for i in range(15):
        Ie_Operator(i)