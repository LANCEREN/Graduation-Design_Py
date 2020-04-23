from License_Plate_Chars_Recognize import recognition_train

def Lpcr_Operator(img):
    """进行车牌识别License_Plate_Chars_Recognize"""

    license_num = ""

    # P = TrainLicensePlateClass.ProvinceTrain(iterations=25)
    # P.TrainModel(P.Default_Model)
    # L = TrainLicensePlateClass.LettersTrain(iterations=25)
    # L.TrainModel(L.Default_Model)
    D = recognition_train.DigitsTrain(iterations=25)
    D.PredictImg("/Users/lanceren/Desktop/1.jpg")


if __name__ == "__main__":
    Lpcr_Operator()
