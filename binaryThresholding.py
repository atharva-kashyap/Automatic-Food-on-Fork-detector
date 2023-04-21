import cv2
import os
import os.path
import matplotlib.pyplot as plt

# This is calling the t-test code that Amal wrote and getting the "mask"
# then the mask is getting cleaned up by cropping some parts of the image
# to generate an image that can be used for grabcut

if "__main__" == __name__:
    folderToRead = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/grabCutAfterTtest/"
    folderToSave = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/grabCutAfterTtest/p_thresholded_empty_AfterRect/"
    imageToRead = "p_thresholded_empty.png"

    img = cv2.imread(folderToRead + imageToRead, cv2.IMREAD_UNCHANGED)
    rect = (100, 200, 300, 555)
    y_upper = 100
    y_lower = 480
    x_left = 300
    x_right = 640
    # y-axis: 100 -> 300 and x-axis: 200 -> 555
    print(img.shape) # 480 * 640
    for t in range(100, 200):
        ret, thresh = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
        rows, cols = thresh.shape
        
        # thresh[x_left:x_right, y_upper:y_lower] = 255

        for r in range(rows):
            for c in range(cols):
                if r < y_upper or r > y_lower:
                    thresh[r][c] = 255
                if c < x_left or c > x_right:
                    thresh[r][c] = 255
        cv2.imwrite(folderToSave + "_" + str(t) + ".png", thresh)