from License_Plate_Chars_Recognize import recognition_train
import cv2


def Lpcr_Operator(refined):
    """进行车牌识别License_Plate_Chars_Recognize"""

    license_num = ""
    totalConfidence = 0

    D = recognition_train.DigitsTrain(iterations=25)
    for i, select in enumerate(refined):
        res_pre = D.PredictImg(select)
        license_num += res_pre[0]
        totalConfidence += res_pre[1]
    averageConfidence = totalConfidence/len(refined)
    return license_num, averageConfidence


if __name__ == "__main__":
    D = recognition_train.DigitsTrain(iterations=25)
    img = cv2.imread("/Users/lanceren/Desktop/1.jpg")
    D.PredictImg(img)
    # P = TrainLicensePlateClass.ProvinceTrain(iterations=25)
    # P.TrainModel(P.Default_Model)
    # L = TrainLicensePlateClass.LettersTrain(iterations=25)
    # L.TrainModel(L.Default_Model)