import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

if "__main__" == __name__:
    folder = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/2022_11_29_2_grape_acquisition_success_1_failure/"
    folderToSave = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/2022_11_29_2_grape_acquisition_success_1_failure_DepthAfterCropResults_smallRange/"
    img = cv.imread(folder + "1669767072677991578.png", cv.IMREAD_COLOR)
    depth_img = cv.imread(folder + "1669767072677991578_depth.png", cv.IMREAD_UNCHANGED)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # define the rectangle that we want to focus on
    rect = (200, 100, 555, 340)
    plt.imshow(img)
    plt.show()

    # apply grabcut method
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    cv.imshow("Image", img)
    img = img[100:340, 200:555]
    cv.imshow("Image_crop", img)
    cv.imwrite(folderToSave + "Image_cropped.png", img)

    depth_img = depth_img[100:340, 200:555]
    depth_thresholds = range(220, 350, 1)
    # for d in depth_thresholds:
    for d in depth_thresholds:
        thresh = d
        rows, cols = depth_img.shape
        for r in range(rows):
            for c in range(cols):
                depVal = depth_img[r, c]
                if (depVal > thresh):
                    img[r, c] = 0
                else:
                    img[r, c] = 255
        # cv.imshow("After thresholding Image", actImg)
        cv.imwrite(folderToSave + "AfterThreshold_" + str(thresh) + ".png", img)
        cv.waitKey()
