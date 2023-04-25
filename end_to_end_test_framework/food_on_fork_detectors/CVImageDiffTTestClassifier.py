import scipy

from .FoodOnForkInterface import FoodOnForkInterface
import numpy as np
import cv2


class CVImageDiffTTestClassifier(FoodOnForkInterface):
    def __init__(self,  colorspace_conversion=cv2.COLOR_BGR2HSV, desired_dimensions=[0, 1, 2]):
        """
        Initializes the CV Image Difference T-Test classifier.

        Parameters
        ----------

        """
        # Initialize the random number generator
        self.colorspace_conversion = colorspace_conversion
        self.desired_dimensions = desired_dimensions
        self.prevImg = None
        self.sum_val = None
        self.sum_val_sq = None
        self.num_imgs = None

    def initialize(self):
        """
        Called right before a bite acquisition attempt. This function should
        reset any state that was maintained from the previous bite acquisition
        attempt.
        """
        self.prevImg = None
        self.sum_val = 0
        self.sum_val_sq = 0
        self.num_imgs = 0

    def next_image(self, img):
        """
        Called when the camera captures a new frame in the bite acquisition
        attempt.
        check if there is a previous image
        if there is a prev image, compute diff and that is val
        then, update sum_val and sum_val_sq
        """
        # Update the number of images
        self.num_imgs += 1

        # Will need to convert to HSV
        # Convert the image from RGB to corresponding colorspace
        raw_img = cv2.cvtColor(img, self.colorspace_conversion)

        # To represent the third element (RGB) for the raw_img, we need to do the following
        # In this desired_dimensions are representative of 0, 1, 2, where 0 is Red, 1 is green, and 2 is Blue
        raw_img = raw_img[:, :, self.desired_dimensions]

        if self.prevImg is None:
            self.prevImg = raw_img
            return

        # Calculate abs diff (if prevImg and raw_img are the same, then absdiff would be 0 otherwise the difference!)
        val = cv2.absdiff(raw_img, self.prevImg)
        cv2.imshow("after absdiff", val)
        cv2.waitKey(20)

        # if we are in a single channel, then we want to make sure to expand that to (w, h, k) shape
        # (axis=2 means add another column)
        if len(val.shape) == 2:
            val = np.expand_dims(val, axis=2)

        # Assuming channel independence, which means we are only looking at the diagonal
        # einsum performs -> collapses the 3rd and 4th dimensions (and makes it ijk instead)
        # inorder to set the diagnols to the squared vals
        val_sq = np.zeros((*val.shape, val.shape[-1]))
        np.einsum("ijkk->ijk", val_sq)[...] = val ** 2.0

        # update sum_val & prevImg
        if self.sum_val is None:
            self.sum_val = val
            self.sum_val_sq = val_sq
        else:
            self.sum_val += val
            self.sum_val_sq += val_sq

        self.prevImg = raw_img

    def predict(self):
        """
        Called at the end of a bite acquisition attempt. Should return a
        boolean, whether there is food on the fork or not.
        Lines 93 - 101 - needs to go here
        get_singular_covar_matrixes

        """
        mean = self.sum_val / self.num_imgs
        cv2.imshow("mean", mean)
        cv2.waitKey(0)

        # Assuming channel independence:
        mean_sq = np.zeros((*mean.shape, mean.shape[-1]))
        np.einsum("ijkk->ijk", mean_sq)[...] = mean ** 2.0

        # calculate covariance
        pop_covar = (self.sum_val_sq / self.num_imgs) - mean_sq
        covar = pop_covar * self.num_imgs / (self.num_imgs - 1)

        # get the number of channels
        k = mean.shape[-1]

        # Use per-pixel Hotelling t-square
        indices_to_mask = {}
        indices_to_mask = self.__get_singular_covar_matrices(covar, indices_to_mask)

        # Although less efficient, we have to do the t_sq computation as a nested
        # for loop because some covariance matrices are masked and have less than
        # three channels used, whereas others use the full covariance matrix
        # TODO: Make this code more efficient; perhaps use numpy methods?
        t_sqs = np.zeros(mean.shape[:-1], dtype=np.float64)
        for i in range(covar.shape[0]):
            for j in range(covar.shape[1]):
                # print(i,j)
                covar_ij = covar[i, j, :, :]
                mask = np.ones(covar_ij.shape[0], dtype=bool)
                if (i, j) in indices_to_mask:
                    mask = indices_to_mask[(i, j)]
                try:
                    t_sq = np.matmul(np.expand_dims(mean[i, j, mask] - 0, axis=0),
                                     np.matmul(np.linalg.inv(covar_ij[mask, :][:, mask]),
                                               np.expand_dims(mean[i, j, mask] - 0, axis=1)))
                except Exception as e:
                    print((i, j), mask, covar_ij, covar_ij[mask, :][:, mask])
                    raise Exception()
                t_sqs[i, j] = t_sq

        # TODO: Write comments about the things below!
        F_vals = (self.num_imgs - k) / ((self.num_imgs - 1) * k) * t_sqs
        p = 1 - scipy.stats.f.cdf(F_vals, k, self.num_imgs - k)
        p_uint = ((p - 0) / (1 - 0) * 255).astype(np.uint8)
        otsu_thresh, thresholded = cv2.threshold(p_uint, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # p value in the end is the image
        img = p
        cv2.imshow("/image_after_t-test.png", img)
        cv2.waitKey(0)

        # Normalizes the image: Changes the range of pixel intensity values
        result = cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply otsu thresholding + cleaning up the edges
        otsu_thresh, thresholded_img = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("/AfterOtsuThresh", thresholded_img)
        cv2.waitKey(0)

        # Use the hand-made mask
        handMadeMask = cv2.imread("//wsl$/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/endToEnd/forktip_food_detection_cv/end_to_end_test_framework/food_on_fork_detectors/Masks/MaskWithCarrot_2_7_23.png", cv2.IMREAD_GRAYSCALE)
        handMadeMask = cv2.erode(handMadeMask, np.ones((50, 50), np.uint8), iterations=1)
        print(handMadeMask)
        cv2.imshow("handmademask", handMadeMask)
        cv2.waitKey(0)

        # apply more cleaning up
        rows, cols = handMadeMask.shape
        for r in range(rows):
            for c in range(cols):
                if handMadeMask[r][c] > 0:
                    thresholded_img[r][c] = 255
        cv2.imshow("imgAfterCleaningUp", thresholded_img)
        cv2.waitKey(0)

        # apply closing (basically looks in the 4x4 matrix and sees if there are any
        # any 0s in that square and if there are, makes the entire 4x4 as 0).
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        cv2.imshow("After Closing", closing)
        cv2.waitKey(0)

        # number of black pixels
        num_black_pixels = np.sum(thresholded_img == 0)
        print(num_black_pixels)
        return True

    def __get_singular_covar_matrices(self, covars, indices_to_mask):
        """
        - Takes in a collection of covariance matrices of size (w, h, k, k).
        - Determines which of those covariance matrices are singular.
        - For the ones that are singular, return a map from their index to the
          channels that are do covary. If a mask already exists for that index,
          take their logical and.
        """
        for i in range(covars.shape[0]):
            for j in range(covars.shape[1]):
                covar = covars[i, j, :, :]
                # If the determinent is 0, the matrix is singular
                if np.isclose(np.linalg.det(covar), 0):
                    # We should mask out the channels that didn't vary, which are
                    # the rows (or columns, since covariance is a triangular matrix)
                    # that are all 0
                    mask = np.logical_not(np.all(np.isclose(covar, 0), axis=1))
                    if (i, j) in indices_to_mask:
                        indices_to_mask[(i, j)] = np.logical_and(indices_to_mask[(i, j)], mask)
                    else:
                        indices_to_mask[(i, j)] = mask
        return indices_to_mask