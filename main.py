import os, sys
import cv2
from pathlib2 import Path
from GUI import gui_operate
from License_Plate_Chars_Recognize import lpcr_operate
from License_Plate_Color_Recognize import lpcor_operate
from License_Plate_Localization import lpl_operate
from License_Plate_Localization import make_data
from Image_Enhancement import ie_operate
from Optical_Char_Recognize import ocr_operate


path = ""
img = cv2.imread(path)
lpl_operate.Lpl_Operator(img)
color = lpcor_operate.Lpcor_Operator(img, path)
ocr_operate.Ocr_Operator(img, path)
lpcr_operate.Lpcr_Operator(img)


# path = r"/Users/lanceren/PycharmProjects/LPR_OpenCV_Graduation/License_Plate_Localization/data/dataset/JPEGImages/ccpd_sample"
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



# gui_operator.Gui_Generator()
# ie_operate.Ie_Operator(1)
