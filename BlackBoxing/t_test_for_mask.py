import cv2 as cv
import math
import numpy as np
import os
import scipy
import traceback

def get_filepaths_sorted_by_ts(images_dir):
    """
    Loads all images in that folder whose extension is PNG and whose name
    is an integer. Returns:
        1) a dictionary that goes from timestamp to filepath
        2) a list that contains all timestamps in sorted order
    """
    filepaths = {} # dictionary from timestamp (int) --> filepath
    for filename in os.listdir(images_dir):
        filepath = os.path.join(images_dir, filename)
        if os.path.isfile(filepath) and "png" in filename.lower():
            try:
                timestamp = int(filename[:-4])
                filepaths[timestamp] = filepath
            except Exception as e:
                traceback.print_exc()
                print(e, "SKIPPING FILE %s" % filepath)
                continue

    ts_sorted = list(sorted(filepaths.keys()))

    return filepaths, ts_sorted

def compute_mean_and_covariance(filepaths, ts_sorted, colorspace_conversion,
    start_i, end_i, diff=False, assume_channel_independence=False, desired_dimensions=[0,1,2]):
    """
    Computes the per-pixel mean and covariance for every image between
    start_i and end_i (inclusive).

    If diff, it first takes the absdiff of each sequential pair of images and
    outputs the mean and covariance of that.
    """

    # Accumulators to compute mean and covariance
    sum_val = None # (w, h)
    sum_val_sq = None # (w, h, k, k)
    num_imgs = end_i - start_i + 1

    # img_temp = []
    prev_raw_img = None
    for i in range(start_i, end_i+1):
        filepath = filepaths[ts_sorted[i]]

        # Load image
        raw_img = cv.imread(filepath, cv.IMREAD_COLOR)
        # Colorspace conversion
        raw_img = cv.cvtColor(raw_img, colorspace_conversion)
        raw_img = raw_img[:,:,desired_dimensions]
        # # Gaussian smoothening
        # raw_img = cv.GaussianBlur(raw_img, (7, 7), 0)
        # cv.imshow("img", img.astype(np.uint8))
        # cv.waitKey(0)
        print(raw_img[220, 460])

        if diff:
            if prev_raw_img is None:
                prev_raw_img = raw_img
                continue
            val = cv.absdiff(raw_img, prev_raw_img)
        else:
            val = raw_img
        val = val.astype(np.float64)

        # If one-channel, make that channel explicit.
        # After this, img will be of shape (w, h, k) where k is the num channels
        if len(val.shape) == 2:
            val = np.expand_dims(val, axis=2)

        # img_temp.append(img)

        # Add to mean and covariance accumulators
        if assume_channel_independence:
            val_sq = np.zeros((*val.shape, val.shape[-1]))
            np.einsum("ijkk->ijk", val_sq)[...] = val**2.0
        else:
            val_sq = np.matmul(np.expand_dims(val, axis=3), np.expand_dims(val, axis=2))
        if sum_val is None:
            sum_val = val
            sum_val_sq = val_sq
        else:
            sum_val += val
            sum_val_sq += val_sq

        prev_raw_img = raw_img

    mean = sum_val / num_imgs
    if assume_channel_independence:
        mean_sq = np.zeros((*mean.shape, mean.shape[-1]))
        np.einsum("ijkk->ijk", mean_sq)[...] = mean**2.0
    else:
        mean_sq = np.matmul(np.expand_dims(mean, axis=3), np.expand_dims(mean, axis=2))
    pop_covar = (sum_val_sq / num_imgs) - mean_sq
    # Convert to sample covariance
    covar = pop_covar * num_imgs / (num_imgs - 1)
    # raise Exception()

    return mean, covar

def get_singular_covar_matrices(covars, indices_to_mask):
    """
    - Takes in a collection of covariance matrices of size (w, h, k, k).
    - Determines which of those covariance matrices are singular.
    - For the ones that are singular, return a map from their index to the
      channels that are do covary. If a mask already exists for that index,
      take their logical and.
    """
    for i in range(covars.shape[0]):
        for j in range(covars.shape[1]):
            covar = covars[i,j,:,:]
            # If the determinent is 0, the matrix is singular
            if np.isclose(np.linalg.det(covar), 0):
                # We should mask out the channels that didn't vary, which are
                # the rows (or columns, since covariance is a triangular matrix)
                # that are all 0
                mask = np.logical_not(np.all(np.isclose(covar, 0), axis=1))
                if (i,j) in indices_to_mask:
                    indices_to_mask[(i,j)] = np.logical_and(indices_to_mask[(i,j)], mask)
                else:
                    indices_to_mask[(i,j)] = mask
    return indices_to_mask

def get_per_pixel_difference(filepaths, ts_sorted, colorspace_conversion, pre_imshow_colorspace_conversion,
    desired_dimensions, in_start_ts, in_end_ts, out_start_ts, out_end_ts, suffix, view_zero_var_indices=False):
    """
    Computes the per-pixel per-channel mean and covariance over the in motion and the
    out motion. Then runs a two-sample t-test to check for difference, and outputs
    a grayscale image with confidence values (p values).
    """
    # get the indexes of the start image + end image
    out_start_i = ts_sorted.index(out_start_ts)
    out_end_i = ts_sorted.index(out_end_ts)

    # get the difference in indexes
    out_n = out_end_i - out_start_i + 1

    # computes the pixel-level mean, covariance
    out_mean, out_covariance = compute_mean_and_covariance(filepaths, ts_sorted,
        colorspace_conversion, out_start_i, out_end_i, diff=True, assume_channel_independence=True,
        desired_dimensions=desired_dimensions)


    k = out_mean.shape[-1] # num channels
    out_mean_viz = out_mean.astype(np.uint8)
    out_mean_viz[220, 460, :] = 255
    # cv.imshow("out_mean", cv.cvtColor(out_mean_viz, pre_imshow_colorspace_conversion))
    # cv.waitKey(0)
    # raise Exception()

    # View the masked indices
    if view_zero_var_indices:
        img = cv.imread(filepaths[ts_sorted[in_end_i]], cv.IMREAD_COLOR)
        # Colorspace conversion
        img = cv.cvtColor(img, colorspace_conversion)
        # Gaussian smoothening
        img = cv.GaussianBlur(img, (7, 7), 0)
        for i, j in indices_to_mask:
            img[i, j, :] = 255
        cv.imshow("img", cv.cvtColor(img, pre_imshow_colorspace_conversion))
        cv.waitKey(0)

    # Compute the per-pixel Hotelling t-sq value
    # https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Hotelling_t-squared_statistic
    # If one channel of a pixel did not change at all, it is possible that the
    # covariance matrix is singular. In those cases, we want to use the
    # covariance matrix for only the channels that did vary.
    indices_to_mask = {}
    indices_to_mask = get_singular_covar_matrices(out_covariance, indices_to_mask)

    # Although less efficient, we have to do the t_sq computation as a nested
    # for loop because some covariance matrices are masked and have less than
    # three channels used, whereas others use the full covariance matrix
    t_sqs = np.zeros(out_mean.shape[:-1], dtype=np.float64)
    for i in range(out_covariance.shape[0]):
        for j in range(out_covariance.shape[1]):
            # print(i,j)
            covar_ij = out_covariance[i,j,:,:]
            mask = np.ones(covar_ij.shape[0], dtype=bool)
            if (i,j) in indices_to_mask:
                mask = indices_to_mask[(i,j)]
            try:
                t_sq = np.matmul(np.expand_dims(out_mean[i,j,mask] - 0, axis=0), np.matmul(np.linalg.inv(covar_ij[mask,:][:,mask]), np.expand_dims(out_mean[i,j,mask] - 0, axis=1)))
            except Exception as e:
                print((i,j), mask, covar_ij, covar_ij[mask,:][:,mask])
                raise Exception()
            t_sqs[i,j] = t_sq
    # t_sqs = np.matmul(np.expand_dims(out_mean - 0, axis=0), np.matmul(np.linalg.inv(out_covariance), np.expand_dims(out_mean - 0, axis=1)))
    F_vals = (out_n - k)/((out_n - 1)*k) * t_sqs
    p = 1-scipy.stats.f.cdf(F_vals, k, out_n - k)
    print("out", out_mean[220, 460], out_covariance[220, 460])
    print("stats", t_sqs[220, 460], F_vals[220, 460], p[220, 460])
    # cv.imshow("p_thresholded", p)
    # cv.waitKey(0)

    # kernel = np.ones((3,3),np.uint8)
    kernel = np.zeros((3,3),np.uint8)
    kernel[0,1] = 1
    kernel[1,0] = 1
    kernel[1,1] = 1
    kernel[1,2] = 1
    kernel[0,2] = 1
    for thresh in np.arange(0, 1.01, 0.01):
        print(thresh)
        p_thresholded = np.where(p <= thresh, 0, 255).astype(np.uint8)
        # # Fill Holes
        # p_thresholded = cv.morphologyEx(p_thresholded, cv.MORPH_CLOSE, kernel)
        cv.imshow("p_thresholded", p_thresholded)
        if thresh >= 0.4 and thresh <= 0.5:
            cv.waitKey(50)
        else:
            cv.waitKey(50)
    p_uint = ((p - 0)/(1 - 0)*255).astype(np.uint8)
    otsu_thresh, thresholded = cv.threshold(p_uint, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    print(otsu_thresh) # 166/255 = 0.65
    cv.imshow("p_thresholded", thresholded)
    cv.waitKey(0)

    # added here:
    return p;

    # raise Exception(p)

if __name__ == "__main__":
    # acquisition_trials = [ # (images_dir, colorspace_conversion, pre_imshow_colorspace_conversion, start skewering timestamp, first experience force bump timestamp, start lifting timestamp, end lifting timestamp, suffix)
    #     ("//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/ada_camera_all_images_compressed_bagfile/",
    #         cv.COLOR_RGB2HSV, cv.COLOR_HSV2BGR, [0,1,2], None, None, 1658882621275864528,1658882630564851082,
    #         "unsuccesful_acquisition_wire_in_the_way")
        # ("//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/ada_camera_all_images_compressed_bagfile/",
        #     cv.COLOR_RGB2HSV, cv.COLOR_HSV2BGR, [0,1,2], 1667339146808778240, 1667339147622624768, 1667339148347074816, 1667339150241960448,
        #     "succesful_acquisition_wire_not_in_the_way"),
        # ("//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/ada_camera_all_images_compressed_bagfile/",
        #     cv.COLOR_RGB2HSV, cv.COLOR_HSV2BGR, [0], None, None, 1667339196380525056,1667339198420471296,
        #     "unsuccesful_acquisition_wire_in_the_way"),
        # ("//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/ada_camera_all_images_compressed_bagfile/",
        #     cv.COLOR_RGB2HSV, cv.COLOR_HSV2BGR, [0], 1667339146808778240, 1667339147622624768, 1667339148347074816, 1667339150241960448,
        #     "succesful_acquisition_wire_not_in_the_way"),
    # ]

    for images_dir, colorspace_conversion, pre_imshow_colorspace_conversion, desired_dimensions, in_start_ts, in_end_ts, out_start_ts, out_end_ts, suffix in acquisition_trials:

        filepaths, ts_sorted = get_filepaths_sorted_by_ts(images_dir)

        get_per_pixel_difference(filepaths, ts_sorted, colorspace_conversion, pre_imshow_colorspace_conversion, desired_dimensions, in_start_ts, in_end_ts, out_start_ts, out_end_ts, suffix)
