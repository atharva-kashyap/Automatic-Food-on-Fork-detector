import cv2 as cv
import math
import numpy as np
import os
import scipy
import traceback
from matplotlib import pyplot as plt
from t_test_for_mask import *

def applyCrop(readImagePath, saveImagePath):
    img = cv.imread(readImagePath, cv.IMREAD_COLOR)
    cropped_img = img[50:600, 250:500]
    cv.imshow("croppedImage", cropped_img)
    cv.imwrite(saveImagePath + "croppedImage.png", cropped_img)
    cv.waitKey(0)


def applyGrabCut(readImagePath, saveImagePath):
    img = cv.imread(readImagePath, cv.IMREAD_COLOR)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # define the rectangle that we want to focus on
    rect = (50, 50, 220, 428)
    plt.imshow(img)
    plt.show()

    # apply grabcut method with RECTANGLE
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    cv.imshow("Image1", img)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    cv.imshow("Image2", img)
    # img = img[100:300, 200:555]
    # cv.imshow("Image_crop", img)
    cv.waitKey(0)

    # apply grabcut method with MASK
    # mask, bgdModel, fgdModel = cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    # mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # img = img * mask[:, :, np.newaxis]
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()

    # convert to a type to be able to save
    result = cv.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imwrite(saveImagePath + "Image_cropped.png", result)


if __name__ == '__main__':
    imageFolder = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/ada_camera_all_images_compressed_bagfile/"
    saveFolder = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/grabCutAfterTtest/"
    # initial time (when food is on fork)
    # startTime = 1658882621275864528
    # endTime = 1658882630564851082

    # when fork is empty
    # startTime = 1658882656199493294
    # endTime = 1658882664532095202

    imageFolder = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/2022_11_01_ada_picks_up_carrots_camera_ft_tf/"
    saveFolder = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/grabCutAfterTtest/"
    startTime = 1667340043034969027
    endTime = 1667340046225560565

    filepaths, ts_sorted = get_filepaths_sorted_by_ts(imageFolder)
    img = get_per_pixel_difference(filepaths, ts_sorted, cv.COLOR_RGB2HSV, cv.COLOR_HSV2BGR, [0,1,2], None, None, startTime, endTime,
            "p_thresholded_2022_11_01_ada_picks_up_carrots_camera_ft_tf")

    cv.imshow(saveFolder + "p_thresholded_empty_2022_11_01_ada_picks_up_carrots_camera_ft_tf.png", img)

    result = cv.normalize(img, dst=None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imwrite(saveFolder + "p_thresholded_empty_2022_11_01_ada_picks_up_carrots_camera_ft_tf.png", result)

    cv.waitKey(0);

    applyCrop(saveFolder + "p_thresholded_empty_2022_11_01_ada_picks_up_carrots_camera_ft_tf.png", saveFolder)
    # applyGrabCut(saveFolder + "croppedImage_empty.png", saveFolder)