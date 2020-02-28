import cv2
import Global_Var
from Image_Enhancement import DarkChannelPrior

def ie_operator():

    m = DarkChannelPrior.deHaze(cv2.imread(Global_Var.Project_Path + '/Source_Pict/wu.jpg') / 255.0) * 255
    cv2.imwrite(Global_Var.Project_Path + '/Image_Enhancement/IE_DataSet/1.png', m)
