import cv2
from pathlib2 import Path
from global_var import globalVars
from Image_Enhancement import image_enhance_process


def Ie_Operator(i):
    inputPath = globalVars.projectPath / Path('Image_Enhancement', 'data', 'dataset', f'{i}.jpg')
    outputPath = globalVars.projectPath / Path('Image_Enhancement', 'data', 'dataset', f'{i}_.png')

    m = image_enhance_process.deHaze(cv2.imread(inputPath.__str__()) / 255.0) * 255
    cv2.imwrite(outputPath.__str__(), m)


if __name__ == "__main__":
    Ie_Operator(2)