import cv2 as cv
import math
import numpy as np
import os

if __name__ == "__main__":
    images_dir = "/Users/amaln/Documents/PRL/ada_amal/automated_bite_detection/2022_11_01_ada_picks_up_carrots_camera_compressed_ft_tf"
    append_suffix = "_all_pairs" # "" #
    # start_time, end_time, suffix = 1667339206239848448, 1667339208486266112, "succesful_acquisition_wire_in_the_way%s" % append_suffix
    start_time, end_time, suffix = 1667339148347074816, 1667339150241960448, "succesful_acquisition_wire_not_in_the_way%s" % append_suffix
    # start_time, end_time, suffix = 1667339196380525056,1667339198420471296, "unsuccesful_acquisition_wire_in_the_way%s" % append_suffix

    filepaths = {} # dictionary from timestamp (int) --> filepath
    for filename in os.listdir(images_dir):
        filepath = os.path.join(images_dir, filename)
        if os.path.isfile(filepath) and "png" in filename.lower():
            timestamp = int(filename[:-4])
            filepaths[timestamp] = filepath

    ts_sorted = list(sorted(filepaths.keys()))
    start_i = ts_sorted.index(start_time)
    end_i = ts_sorted.index(end_time)
    num_images = end_i - start_i + 1
    weight = 1.0/num_images

    ############################################################################
    # SEQUENTIAL DIFFERENCE
    ############################################################################

    composite_img = None
    prev_img = None
    for i in range(start_i, end_i+1):
        filepath = filepaths[ts_sorted[i]]
        # img = cv.imread(filepath, cv.IMREAD_COLOR)
        img = cv.cvtColor(cv.imread(filepath, cv.IMREAD_COLOR), cv.COLOR_RGB2BGR)

        # Gaussian Smoothening
        img = cv.GaussianBlur(img, (7, 7), 0)

        if prev_img is not None:
            # Compute the difference
            diff = cv.cvtColor(cv.absdiff(img, prev_img), cv.COLOR_RGB2GRAY)
            # cv.imshow('window', diff)
            # cv.waitKey(150)
            # print(np.quantile(diff, 1.0), np.quantile(diff, 0.75), np.quantile(diff, 0.5), np.quantile(diff, 0.25), np.quantile(diff, 0.0))


            # Run Otsu Thresholding on the difference
            otsu_thresh, thresholded = cv.threshold(diff, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            print(otsu_thresh)
            cv.imshow('window', thresholded)
            cv.waitKey(200)

            # print("A", np.count_nonzero(thresholded==255), np.count_nonzero(thresholded==0))

        # # Display the images
        # cv.imshow('window', img)
        # cv.waitKey(50)

        # # The composite image is an average of all images
        # if composite_img is None:
        #     composite_img = img.astype(float)*weight
        # else:
        #     composite_img += img*weight

        # The composite image is an accumulation of differences
        if prev_img is not None and otsu_thresh >= 5.0:
            if composite_img is None:
                composite_img = img.astype(float)
                # composite_img = thresholded.astype(float)*weight
            else:
                mask_white = (thresholded == 255)
                composite_img[mask_white] = 255
                mask_img = composite_img != 255
                composite_img[mask_img] = img[mask_img]

                # composite_img += thresholded*weight

        # Increment the previous image
        prev_img = img

    composite_img = np.uint8(composite_img)
    # cv.imshow('window', composite_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    cv.imwrite(os.path.join(images_dir, "../%s_%d_%d_%s.png" % (images_dir.split("/")[-1], start_time, end_time, suffix)), composite_img)

    # ############################################################################
    # # ALL PAIRS DIFFERENCE
    # ############################################################################
    #
    # i_to_img = {}
    # composite_img = None
    # for i in range(start_i, end_i+1):
    #     if i in i_to_img:
    #         img = i_to_img[i]
    #     else:
    #         # img = cv.imread(filepath, cv.IMREAD_COLOR)
    #         img = cv.cvtColor(cv.imread(filepaths[ts_sorted[i]], cv.IMREAD_COLOR), cv.COLOR_RGB2BGR)
    #
    #         # Gaussian Smoothening
    #         img = cv.GaussianBlur(img, (7, 7), 0)
    #
    #         i_to_img[i] = img
    #
    #     for j in range(i, end_i+1):
    #         if j in i_to_img:
    #             comp_img = i_to_img[j]
    #         else:
    #             # img = cv.imread(filepath, cv.IMREAD_COLOR)
    #             comp_img = cv.cvtColor(cv.imread(filepaths[ts_sorted[j]], cv.IMREAD_COLOR), cv.COLOR_RGB2BGR)
    #
    #             # Gaussian Smoothening
    #             comp_img = cv.GaussianBlur(comp_img, (7, 7), 0)
    #
    #             i_to_img[j] = comp_img
    #
    #             diff = cv.cvtColor(cv.absdiff(img, comp_img), cv.COLOR_RGB2GRAY)
    #
    #
    #             # Run Otsu Thresholding on the difference
    #             otsu_thresh, thresholded = cv.threshold(diff, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #
    #             # Compute the difference
    #             diff = cv.cvtColor(cv.absdiff(img, comp_img), cv.COLOR_RGB2GRAY)
    #             # cv.imshow('window', diff)
    #             # cv.waitKey(0)
    #             # print(np.quantile(diff, 1.0), np.quantile(diff, 0.75), np.quantile(diff, 0.5), np.quantile(diff, 0.25), np.quantile(diff, 0.0))
    #
    #
    #             # Run Otsu Thresholding on the difference
    #             otsu_thresh, thresholded = cv.threshold(diff, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #
    #             if otsu_thresh >= 5.0:
    #                 if composite_img is None:
    #                     composite_img = comp_img.astype(float)
    #                     # composite_img = thresholded.astype(float)*weight
    #                 else:
    #                     mask_white = (thresholded == 255)
    #                     composite_img[mask_white] = 255
    #                     mask_img = composite_img != 255
    #                     composite_img[mask_img] = comp_img[mask_img]

    composite_img = np.uint8(composite_img)
    cv.imshow('window', composite_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite(os.path.join(images_dir, "../%s_%d_%d_%s.png" % (images_dir.split("/")[-1], start_time, end_time, suffix)), composite_img)
