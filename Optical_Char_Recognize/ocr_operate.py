from Optical_Char_Recognize import optical_char_recognize_process
import cv2
import os
from pathlib2 import Path


def Ocr_Operator(img, fileName):
    cvtimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    refined, score, name, averageConfidence = optical_char_recognize_process.slidingWindowsEval(cvtimg, fileName)
    return refined, score, name, averageConfidence


if __name__ == "__main__":
    path = "/Users/lanceren/Desktop/川A2SV59.jpg"
    img = cv2.imread(path)
    cvtimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a, b, c, d = optical_char_recognize_process.slidingWindowsEval(cvtimg, "川A2SV59")

    #
    # path = r"/Users/lanceren/Downloads/GD_Dataset/Raw_Data/车牌检测数据/训练车牌检测模型数据/has"
    # for rt, dirs, files in os.walk(path):
    #     files = [f for f in files if not f[0] == '.']
    #     dirs[:] = [d for d in dirs if not d[0] == '.']
    #     for fileName in files:
    #         fullFileName = Path(rt, fileName)
    #         img = cv2.imread(fullFileName.__str__())
    #         cvtimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         a, b, c = optical_char_recognize_process.slidingWindowsEval(cvtimg, fileName)
    pass
