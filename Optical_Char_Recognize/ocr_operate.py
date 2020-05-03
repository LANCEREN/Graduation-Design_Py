from Optical_Char_Recognize import optical_char_recognize_process
import cv2
import os
from pathlib2 import Path


def Ocr_Operator(img, fileName):
    cvtimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    refined, score = optical_char_recognize_process.slidingWindowsEval(cvtimg, fileName)
    return refined, score


if __name__ == "__main__":

    pass
