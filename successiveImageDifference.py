import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

def calculateAbsDiff(t0img, t1img):
    # Load images as grayscale
    t_0_image = cv.imread(t0img, cv.IMREAD_GRAYSCALE)
    t_1_image = cv.imread(t1img, cv.IMREAD_GRAYSCALE)

    # Calculate the per-element absolute difference between
    # two arrays or between an array and a scalar
    diff = 255 - cv.absdiff(t_0_image, t_1_image)

    # show the difference
    # cv.imshow('diff', diff)
    # save the difference
    # cv.imwrite("difference.jpg", diff)
    # for any window that is displayed, you want to wait for the window to stay open
    # cv.waitKey()
    return diff

def calculateAbsDiffOverInterval(startTime, endTime, interval, foldername):
    runningTotal = None
    count = 0
    runningPartialSD = None
    runningSD = 0
    mean = None
    variance = None
    fileList = os.listdir(foldername)
    for i in fileList:
        t0img = foldername + str(i)
        t1img = foldername + fileList[fileList.index(str(i)) + 1]
        difference = calculateAbsDiff(t0img, t1img)
        if runningTotal is not None:
            runningTotal += difference
        else:
            runningTotal = difference
        count += 1
        mean = runningTotal / count
        if runningPartialSD is not None:
            runningPartialSD += (difference - mean)**2
        else:
            runningPartialSD = (difference - mean)**2
        runningSD += runningPartialSD / count
        variance = runningSD**2

    print(mean)
    cv.imshow("Mean", mean)
    cv.waitKey()
    print(variance)
    cv.imshow("Variance", variance)
    cv.waitKey()



if "__main__" == __name__:
    foldername = "//wsl.localhost/Ubuntu/home/atharvak/cv2Workspace/src/images/ada_camera_all_images_compressed_bagfile/"
    calculateAbsDiffOverInterval(1658882621441155564, 1658882627373575265, 27000000, foldername)