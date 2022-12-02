# success: rgb -> otsu -> 1) grayscale; 2) r; 3) g; 4) b
# success: hsv -> otsu -> 1) grayscale; 2) h; 3) s; 4) v
# success: yuv -> otsu -> 1) grayscale; 2) y; 3) u; 4) v
# failure: rgb -> otsu -> 1) grayscale; 2) r; 3) g; 4) b
# failure: hsv -> otsu -> 1) grayscale; 2) h; 3) s; 4) v
# failure: yuv -> otsu -> 1) grayscale; 2) y; 3) u; 4) v

import cv2 as cv
import numpy as np
import os

def calculateIndex(foldername, startTime, endTime):
    filenameToTimeMap = {}
    fileList = os.listdir(foldername)
    for f in fileList:
        filenameToTimeMap[int(f.strip(".png"))] = f
    sortedFileList = list(sorted(filenameToTimeMap.keys()))
    start_i = sortedFileList.index(startTime)
    end_i = sortedFileList.index(endTime)
    return (sortedFileList, start_i, end_i)

def addOtsuThresholding(img):
    scaledImg = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    print(scaledImg.astype(int))
    ret, thresh = cv.threshold(scaledImg.astype(np.uint8), 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

def calculateAbsDiff(fileList, start_i, end_i, folderToRead, folderToSave, imageReadType, imageConvertType, convertToOneChannel, convertTypeForOtsu, label):
    runningTotal = None
    sumsq = None
    currIndex = start_i
    while currIndex < end_i:
        t0img = folderToRead + str(fileList[currIndex]) + ".png"
        t1img = folderToRead + str(fileList[currIndex + 1]) + ".png"
        t_0_img = cv.imread(t0img, imageReadType)
        t_1_img = cv.imread(t1img, imageReadType)

        # Preprocess Image (add blur)
        t_0_img = cv.GaussianBlur(t_0_img, (7, 7), 0)
        t_1_img = cv.GaussianBlur(t_1_img, (7, 7), 0)

        # cv.imshow("t0img", t_0_img)
        # cv.imshow("t1img", t_1_img)
        # cv.waitKey(0)

        # check if need to convert Image
        if imageConvertType is not None:
            t_0_img = cv.cvtColor(t_0_img, imageConvertType)
            t_1_img = cv.cvtColor(t_1_img, imageConvertType)

        # Calculate Abs Diff
        # In grayscale, will be white if there is a change
        diff = cv.absdiff(t_0_img, t_1_img)

        if runningTotal is not None:
            runningTotal += diff
            sumsq += runningTotal ** 2
        else:
            runningTotal = diff
            sumsq = runningTotal ** 2
        currIndex += 1
        # print(runningTotal / (currIndex - start_i))

    # Regular Means and Variances
    mean = runningTotal / (end_i - start_i)
    meanW = (mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255
    meanW = meanW.astype(np.uint8)
    cv.imshow(label + "/Mean", mean)
    cv.imwrite(folderToSave + label + "/Mean.png", meanW)
    var = (sumsq / (end_i - start_i)) - mean ** 2
    varW = (var - np.min(var)) / (np.max(var) - np.min(var)) * 255
    varW = varW.astype(np.uint8)
    cv.imshow(label + "/Variance", var)
    cv.imwrite(folderToSave + label + "/Variance.png", varW)

    # Otsu Thresholding
    if not convertToOneChannel:
        cv.imshow(label + "/Variance_AfterOtsu", addOtsuThresholding(var))
        cv.imwrite(folderToSave + label + "/Variance_AfterOtsu.png", addOtsuThresholding(var))
    else:
        var_channel1, var_channel2, var_channel3 = cv.split(var)
        cv.imshow(label + "/Variance_channel1_AfterOtsu", addOtsuThresholding(var_channel1))
        cv.imwrite(folderToSave + label + "/Variance_channel1_AfterOtsu.png", addOtsuThresholding(var_channel1))
        cv.imshow(label + "/Variance_channel2_AfterOtsu", addOtsuThresholding(var_channel2))
        cv.imwrite(folderToSave + label + "/Variance_channel2_AfterOtsu.png", addOtsuThresholding(var_channel2))
        cv.imshow(label + "/Variance_channel3_AfterOtsu", addOtsuThresholding(var_channel3))
        cv.imwrite(folderToSave + label + "/Variance_channel3_AfterOtsu.png", addOtsuThresholding(var_channel3))
        if convertTypeForOtsu is not None:
            con1, con2 = convertTypeForOtsu
            if con2 is None:
                varAfterGray = cv.cvtColor(np.float32(var), con1)
                cv.imshow(label + "/Variance_OverallGray_AfterOtsu", addOtsuThresholding(varAfterGray))
                cv.imwrite(folderToSave + label + "/Variance_OverallGray_AfterOtsu.png", addOtsuThresholding(varAfterGray))
            else:
                varAfterCon1 = cv.cvtColor(np.float32(var), con1)
                varAfterCon2 = cv.cvtColor(varAfterCon1, con2)
                cv.imshow(label + "/Variance_OverallGray_AfterOtsu", addOtsuThresholding(varAfterCon2))
                cv.imwrite(folderToSave + label + "/Variance_OverallGray_AfterOtsu.png", addOtsuThresholding(varAfterCon2))

    cv.waitKey(0)

if "__main__" == __name__:
    foldername = "//wsl.localhost/Ubuntu/home/atharvak/cv2Workspace/src/images/ada_camera_all_images_compressed_bagfile/"
    folderToSave = "//wsl.localhost/Ubuntu/home/atharvak/cv2Workspace/src/images/ada_camera_all_images_compressed_bagfile_Results/"

    # (imageReadType, imageConvertType, needsToConvertToOneChannel?, convertTypeForOtsu)
    convertFlags = {"gray": (cv.IMREAD_GRAYSCALE, None, False, None),
                    "rgb": (cv.IMREAD_COLOR, None, True, (cv.COLOR_RGB2GRAY, None)),
                    "hsv": (cv.IMREAD_COLOR, cv.COLOR_RGB2HSV, True, (cv.COLOR_HSV2RGB, cv.COLOR_RGB2GRAY)),
                    "yuv": (cv.IMREAD_COLOR, cv.COLOR_RGB2YUV, True, (cv.COLOR_YUV2RGB, cv.COLOR_RGB2GRAY))}

    foodMeasure = {"success": (1658882621441155564, 1658882627373575265, 27000000),
                   "failure": (1658882561589673588, 1658882565489796245, 27000000)}

    for foodMeasureKey in foodMeasure:
        startTime, endTime, interval = foodMeasure[foodMeasureKey]
        fileList, start_i, end_i = calculateIndex(foldername, startTime, endTime)
        for convertFlagsKey in convertFlags:
            imageReadType, imageConvertType, convertToOneChannel, convertTypeForOtsu = convertFlags[convertFlagsKey]
            calculateAbsDiff(fileList, start_i, end_i,
                             foldername, folderToSave, imageReadType, imageConvertType, convertToOneChannel, convertTypeForOtsu,
                             convertFlagsKey + "_" + foodMeasureKey)
