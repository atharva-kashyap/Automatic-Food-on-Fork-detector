import cv2 as cv
import math
import numpy as np
import os
import scipy
import traceback
from t_test_for_mask import *

"""
    Steps: 
    - First, get the sequence of images (first for a sequence that has a large background change)
    - Use the T-test method to get a matrix of p-values
    - Then use Binary Thresholding on the matrix of p-values generated -> this will generate a mask
    - Clean up the mask
    - Finally, apply the mask on the T-test p-values image
"""

def cleaningUp(img, mask):
    print(mask.shape)
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r][c] > 0:
                img[r][c] = 255
    return img

if "__main__" == __name__:
    folderToRead = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/"
    imageFolder = "2022_11_29_mostly_grape_failures_some_successes_Images/"
    maskLocation = "Masks/MaskWithCarrot_2_7_23.png"
    saveFolder = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/grabCutAfterTtest/blackbox/"

    descriptiveName = "grapefailure_SmallShiftInBG_2022_11_29"

    actualSaveFolder = saveFolder + descriptiveName
    if not os.path.isdir(actualSaveFolder):
        os.mkdir(actualSaveFolder)

    startTime = 1669765759168663283
    endTime = 1669765760553397520

    # Sort the filepaths
    filepaths, ts_sorted = get_filepaths_sorted_by_ts(folderToRead + imageFolder)

    # Apply T-test to get p_thresholded values
    img = get_per_pixel_difference(filepaths=filepaths, ts_sorted=ts_sorted, colorspace_conversion=cv.COLOR_RGB2HSV,
                                   pre_imshow_colorspace_conversion=cv.COLOR_HSV2BGR, desired_dimensions=[0, 1, 2],
                                   in_start_ts=None, in_end_ts=None, out_start_ts=startTime, out_end_ts=endTime,
                                   suffix=descriptiveName)

    cv.imshow(actualSaveFolder + "/image_after_t-test.png", img)
    cv.waitKey(0)
    cv.imwrite(actualSaveFolder + "/image_after_t-test.png", img)

    # Normalizes the image: Changes the range of pixel intensity values
    result = cv.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imwrite(actualSaveFolder + "/after_normalizing.png", result)

    # Apply binary thresholding + clean up the edges
    # ret, binResult = cv.threshold(result, 162, 255, cv.THRESH_BINARY)
    otsu_thresh, thresholded = cv.threshold(result, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow(actualSaveFolder + "/AfterOtsuThresh", thresholded)
    cv.waitKey(0)
    cv.imwrite(actualSaveFolder+ "/AfterOtsuThresh.png", thresholded)

    # Use Hand-made mask
    handMadeMask = cv.imread(folderToRead + maskLocation, cv.IMREAD_GRAYSCALE)
    handMadeMask = cv.erode(handMadeMask, np.ones((50, 50), np.uint8), iterations = 1)
    cv.imshow("handmademask", handMadeMask)
    cv.waitKey(0)

    # apply cleaning up
    img = cleaningUp(thresholded, handMadeMask)
    cv.imshow("imgAfter", img)
    cv.waitKey(0)
    cv.imwrite(actualSaveFolder + "/image_at_the_end.png", img)

    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((4, 4), np.uint8))
    cv.imshow("After Closing", closing)
    cv.waitKey(0)
    cv.imwrite(actualSaveFolder + "/afterclosing.png", closing)

    num_black_pixels = np.sum(img == 0)
    print(num_black_pixels)