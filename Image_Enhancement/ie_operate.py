import cv2
import global_var
from Image_Enhancement import image_enhance_process


def Ie_Operator(i):

    m = image_enhance_process.deHaze(cv2.imread(global_var.projectPath + f'/Image_Enhancement/data/dataset/{i}.jpg') / 255.0) * 255
    cv2.imwrite(global_var.projectPath + f'/Image_Enhancement/data/dataset/{i}_.png', m)


if __name__ == "__main__":
    for i in range(15):
        Ie_Operator(i)