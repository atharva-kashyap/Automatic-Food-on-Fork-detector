# success: rgb -> otsu -> 1) grayscale; 2) r; 3) g; 4) b
# success: hsv -> otsu -> 1) grayscale; 2) h; 3) s; 4) v
# success: yuv -> otsu -> 1) grayscale; 2) y; 3) u; 4) v
# failure: rgb -> otsu -> 1) grayscale; 2) r; 3) g; 4) b
# failure: hsv -> otsu -> 1) grayscale; 2) h; 3) s; 4) v
# failure: yuv -> otsu -> 1) grayscale; 2) y; 3) u; 4) v

import cv2 as cv
import numpy as np
import os


def calculateAbsDiffOverInterval(startTime, endTime, interval, foldername, folderToSave, type):
    fileList = os.listdir(foldername)
    startingIndex = fileList.index(str(startTime) + ".png")
    print(startingIndex)
    time2 = startTime + interval
    currIndex = startingIndex
    secondIndex = None
    while currIndex < len(fileList):
        if int(fileList[currIndex].strip(".png")) >= time2:
            secondIndex = currIndex
            break
        currIndex += 1
    indexDiff = secondIndex - startingIndex
    print(indexDiff)
    currIndex = startingIndex
    print(fileList)
    for key in type:
        actualCalculationOfAbsDiff(currIndex, fileList, endTime, indexDiff, folderToSave, key, type[key])

def actualCalculationOfAbsDiff(currIndex, fileList, endTime, indexDiff, folderToSave, readTypeName, readType):
    runningTotal = None
    sumsq = None
    count = 0
    while int(fileList[currIndex].strip(".png")) < endTime and currIndex < len(fileList):
        t0img = foldername + str(fileList[currIndex])
        t1img = foldername + str(fileList[currIndex + indexDiff])
        try:
            # cv.IMREAD_COLOR
            # cv.COLOR_RGB2HSV
            # cv.COLOR_RGB2YUV
            t_0_image = cv.imread(t0img, eval(readType))
            t_1_image = cv.imread(t1img, eval(readType))
            difference = calculateAbsDiff(t_0_image, t_1_image)
            # print(difference)
            # cv.imshow("diff", np.concatenate((t_0_image, t_1_image, difference), axis=1))
            # if currIndex != 573:
            #     cv.waitKey(50)
            # else:
            #     cv.waitKey()
            count += 1
            if runningTotal is not None:
                runningTotal += difference
                sumsq += runningTotal**2
            else:
                runningTotal = difference
                sumsq = runningTotal**2
        except:
            continue
        currIndex += indexDiff

    mean = runningTotal / count
    # To calculate Variance: https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
    var = (sumsq / count) - mean ** 2
    mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean)) * 255
    mean = mean.astype(np.uint8)
    print("mean", mean)
    var = (var - np.min(var)) / (np.max(var) - np.min(var)) * 255
    var = var.astype(np.uint8)
    print("variance", var)
    # os.chdir(folderToSave)
    cv.imshow(readTypeName + ", Mean", mean)
    cv.imwrite(folderToSave + "/" + readTypeName + "/Mean.png", mean)
    cv.imwrite(folderToSave + "/" + readTypeName + "/MeanGray.png",  cv.cvtColor(np.float32(mean), cv.COLOR_RGB2GRAY))
    cv.imshow(readTypeName + ", Variance", var)
    cv.imwrite(folderToSave + "/" + readTypeName + "/Variance.png", var)
    cv.imwrite(folderToSave + "/" + readTypeName + "/VarianceGray.png", cv.cvtColor(np.float32(var), cv.COLOR_RGB2GRAY))
    # cv.COLOR_RGB2GRAY
    # var_afterColor = cv.cvtColor(np.float32(var), cv.COLOR_YUV2RGB)
    if readType == cv.IMREAD_COLOR:
        var_afterGray = cv.cvtColor(np.float32(var), cv.COLOR_RGB2GRAY)
    elif readType == cv.COLOR_RGB2HSV:
        var_afterColor = cv.cvtColor(np.float32(var), cv.COLOR_HSV2RGB)
        var_afterGray = cv.cvtColor(np.float32(var_afterColor), cv.COLOR_RGB2GRAY)
    else:
        var_afterColor = cv.cvtColor(np.float32(var), cv.COLOR_YUV2RGB)
        var_afterGray = cv.cvtColor(np.float32(var_afterColor), cv.COLOR_RGB2GRAY)
    cv.imshow(readTypeName + ", Grayscale after Otsu", addOtsuThresholding(var_afterGray))
    cv.imwrite(folderToSave + "/" + readTypeName + "/grayscaleAfterOtsu.png", addOtsuThresholding(var_afterGray))
    # cv.imshow("converttoGray", var_afterGray)
    var_channel1, var_channel2, var_channel3 = cv.split(var)
    cv.imshow(readTypeName + ", Channel 1 after Otsu", addOtsuThresholding(var_channel1))
    cv.imwrite(folderToSave + "/" + readTypeName + "/Channel1AfterOtsu.png", addOtsuThresholding(var_channel1))
    cv.imshow(readTypeName + ", Channel 2 after Otsu", addOtsuThresholding(var_channel2))
    cv.imwrite(folderToSave + "/" + readTypeName + "/Channel2AfterOtsu.png", addOtsuThresholding(var_channel2))
    cv.imshow(readTypeName + ", Channel 3 after Otsu", addOtsuThresholding(var_channel3))
    cv.imwrite(folderToSave + "/" + readTypeName + "/Channel3AfterOtsu.png", addOtsuThresholding(var_channel3))
    # cv.imshow("Channel 1", var_channel1)
    # cv.imshow("Channel 2", var_channel2)
    # cv.imshow("Channel 3", var_channel3)
    cv.waitKey()

def calculateAbsDiff(t0img, t1img):
    try:
        # preprocessing
        t0img = cv.GaussianBlur(t0img, (7, 7), 0)
        t1img = cv.GaussianBlur(t1img, (7, 7), 0)
        diff = cv.absdiff(t0img, t1img)
    except Exception:
        print(str(Exception))
    return diff

def addOtsuThresholding(img):
    scaledImg = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    print(scaledImg.astype(int))
    ret, thresh = cv.threshold(scaledImg.astype(np.uint8), 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

if "__main__" == __name__:
    foldername = "//wsl.localhost/Ubuntu/home/atharvak/cv2Workspace/src/images/ada_camera_all_images_compressed_bagfile/"
    # rgb_success, rgb_failure, hsv_success, hsv_failure, yuv_success, yuv_failure
    successTypes = {"rgb_success": 'cv.IMREAD_COLOR', "hsv_success": 'cv.COLOR_RGB2HSV', "yuv_success": 'cv.COLOR_RGB2YUV'}
    failureTypes = {"rgb_failure": 'cv.IMREAD_COLOR', "hsv_failure": 'cv.COLOR_RGB2HSV', "yuv_failure": 'cv.COLOR_RGB2YUV'}
    folderToSave = "//wsl.localhost/Ubuntu/home/atharvak/cv2Workspace/src/images/ada_camera_all_images_compressed_bagfile_Results"
    # Food on the fork Range: 1658882621441155564, 1658882627373575265, 27000000
    # Food not on fork Range: 1658882561589673588, 1658882565489796245, 27000000
    foodMeasure = {"success": (successTypes, 1658882621441155564, 1658882627373575265, 27000000),
                   "failure": (failureTypes, 1658882561589673588, 1658882565489796245, 27000000)}
    for key in foodMeasure:
        types, startTime, endTime, interval = foodMeasure[key]
        calculateAbsDiffOverInterval(startTime, endTime, interval, foldername, folderToSave, types)
    print("done")