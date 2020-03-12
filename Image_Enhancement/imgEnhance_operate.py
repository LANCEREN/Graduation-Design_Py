import cv2
import Global_Var
from Image_Enhancement import DarkChannelPrior

def Ie_Operator():

    m = DarkChannelPrior.deHaze(cv2.imread(Global_Var.projectPath + '/Source_Pict/wu.jpg') / 255.0) * 255
    cv2.imwrite(Global_Var.projectPath + '/Image_Enhancement/IE_DataSet/1.png', m)
