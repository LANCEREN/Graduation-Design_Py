# dirPath = r"/Users/lanceren/Desktop/has"
# for rt, dirs, files in os.walk(dirPath):
#     files = [f for f in files if not f[0] == '.']
#     dirs[:] = [d for d in dirs if not d[0] == '.']
#     for file in files:
#         filePath = Path(rt, file)
#         predictResult = trainClass.SimplePredict(filePath, model)
#         print(predictResult)
#         cv2.imwrite(f"/Users/lanceren/PycharmProjects/LPR_OpenCV_Graduation/License_Plate_Color_Recognize/data/dataset/{predictResult}/{fileName}.jpg",img)
