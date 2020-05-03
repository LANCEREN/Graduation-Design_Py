from License_Plate_Chars_Recognize import recognition_train
from License_Plate_Chars_Recognize import recognizer
import cv2


def Lpcr_Operator(refined, plateImgPrecise):
    """进行车牌识别License_Plate_Chars_Recognize"""
    recognizerClass = recognizer.LPR()
    license_num, confidence = recognizerClass.RecognizePlate(plateImage=plateImgPrecise)
    return license_num, confidence


if __name__ == "__main__":
    # license_num = ""
    # totalConfidence = 0
    #
    # D = recognition_train.DigitsTrain(iterations=25)
    # for i, select in enumerate(refined):
    #     res_pre = D.PredictImg(select)
    #     license_num += res_pre[0]
    #     totalConfidence += res_pre[1]
    # averageConfidence = totalConfidence / len(refined)
    # return license_num, averageConfidence
    pass
    # P = TrainLicensePlateClass.ProvinceTrain(iterations=25)
    # P.TrainModel(P.Default_Model)
    # L = TrainLicensePlateClass.LettersTrain(iterations=25)
    # L.TrainModel(L.Default_Model)