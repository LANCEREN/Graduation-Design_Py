from License_Plate_Chars_Recognize import TrainLicensePlateClass

def Lpcr_Operator():
    """进行车牌识别License_Plate_Chars_Recognize"""

    license_num = ""

    P = TrainLicensePlateClass.ProvinceTrain(iterations=25)
    # P.TrainModel(P.Default_Model)
    # L = TrainLicensePlateClass.LettersTrain(iterations=25)
    # L.TrainModel(L.Default_Model)
    # D = TrainLicensePlateClass.DigitsTrain(iterations=25)
    # D.TrainModel(D.Default_Model)


if __name__ == "__main__":
    Lpcr_Operator()
