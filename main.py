import os, sys
import cv2
import argparse
from pathlib2 import Path
# from GUI import gui_operate
from License_Plate_Chars_Recognize import lpcr_operate
from License_Plate_Color_Recognize import lpcor_operate
from License_Plate_Localization import lpl_operate
from License_Plate_Localization import make_data
from Image_Enhancement import ie_operate
from Optical_Char_Recognize import ocr_operate

parser = argparse.ArgumentParser()
parser.description = "GD python part"
parser.prog = "GD_Python"
parser.add_argument('--make_data', action='store_true',
                    default=False,
                    dest='boolean_make_data',
                    help='Switch to make data')
parser.add_argument('--img_process', action='store_true',
                    default=False,
                    dest='boolean_img_process',
                    help='Switch to image process')
parser.add_argument('--file', action='store',
                    type=str,
                    default='',
                    dest='file',
                    help='Input file path')
parser.add_argument('--folder', action='store',
                    type=str,
                    default='',
                    dest='folder',
                    help='Input folder path')
parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

arg = parser.parse_args()


def ImgProcess(imgPath):
    defaultPath = Path("/Users/lanceren/Desktop/test/2.jpg")
    outPath = Path("/Users/lanceren/Desktop/test/output")
    plateImgPrecisePath = outPath / Path("plateImg_precise.jpg")
    plateImgGeneralPath = outPath / Path("plateImg_general.jpg")
    img = cv2.imread(imgPath.__str__())
    fileName = "test"

    plateImg_general, plateImg_precise, plateConf = lpl_operate.Lpl_Operator(img, fileName)
    refined, score, name, averageConfidence = ocr_operate.Ocr_Operator(plateImg_precise, fileName)
    str, con = lpcr_operate.Lpcr_Operator(refined)
    color = lpcor_operate.Lpcor_Operator(img, fileName)

    print(color, name)

    for i, pic in enumerate(refined):
        fullFilePath = outPath / Path(f"{i}.jpg")
        cv2.imwrite(fullFilePath.__str__(), pic)
    cv2.imwrite(plateImgPrecisePath.__str__(), plateImg_precise)
    cv2.imwrite(plateImgGeneralPath.__str__(), plateImg_general)


def MakeData():
    pass
    # path = r"/Users/lanceren/PycharmProjects/Graduation-Design_Py/License_Plate_Localization/data/dataset/JPEGImages/ccpd_sample"
    # count = 0
    # for rt, dirs, files in os.walk(path):
    #     files = [f for f in files if not f[0] == '.']
    #     dirs[:] = [d for d in dirs if not d[0] == '.']
    #     for fileName in files:
    #         fullFileName = Path(rt, fileName)
    #         ccpdName = make_data.CCPDNameParams(fileName)
    #         print(fileName)
    #         print(count)
    #         img = cv2.imread(fullFileName.__str__())
    #         img = img[ccpdName.ymin:ccpdName.ymax, ccpdName.xmin:ccpdName.xmax]
    #         cvtimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         ocr_operate.Ocr_Operator(cvtimg, fileName)
    #         lpcor_operate.Lpcor_Operator(img, fileName)
    #         count += 1


if arg.boolean_make_data:
    MakeData()
if arg.boolean_img_process:
    ImgProcess(Path(arg.file))
