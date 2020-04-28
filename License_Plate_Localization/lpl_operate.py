from License_Plate_Localization import location_train
from License_Plate_Localization import location_test
from License_Plate_Localization import make_data
import cv2


def Lpl_Operator(img, fileName):
    trainClass = location_train.Train()
    plateImg_whole, plateImg_general, plateImg_precise, plateConf = trainClass.predictImg(img, fileName)
    return [plateImg_whole, plateImg_general, plateImg_precise, plateConf]

if __name__ == "__main__":
    path = "/Users/lanceren/Desktop/A4F7A413-972E-4CF6-B875-B538404556D8_1_105_c.jpg"
    img = cv2.imread(path)
    Lpl_Operator(img, "x")
    pass
