import cv2 as cv
import numpy as np
import os

if "__main__" == __name__:
    foldername = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/2022_11_29_2_grape_acquisition_success_1_failure/"
    folderToSave = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/2022_11_29_2_grape_acquisition_success_1_failure_DepthResults/"

    # read the depth & actual images
    # 1st: 1669765712457254941
    # 2nd: 1669765849531814129
    # New Rosbag: 1) 1669767072677991578
    depthImg = cv.imread(foldername + "1669767072677991578_depth.png", cv.IMREAD_UNCHANGED)
    actImg = cv.imread(foldername + "1669767072677991578.png", cv.IMREAD_COLOR)
    depthImg_scale = (depthImg - np.min(depthImg)) / (np.max(depthImg) - np.min(depthImg)) * 255
    cv.imshow("depImg", depthImg_scale)
    cv.imshow("Before thresholding Image", actImg)

    print("depth Img: ", depthImg)
    print("actual Img: ", actImg)

    depth_thresholds = range(1, 1500, 5)
    # for d in depth_thresholds:
    for d in depth_thresholds:
        thresh = d
        rows, cols = depthImg.shape
        for r in range(rows):
            for c in range(cols):
                depVal = depthImg[r, c]
                if (depVal > thresh):
                    actImg[r, c] = 0
                else:
                    actImg[r, c] = 255
        # cv.imshow("After thresholding Image", actImg)
        cv.imwrite(folderToSave + "AfterThreshold_" + str(thresh) + ".png", actImg)
        cv.waitKey()
