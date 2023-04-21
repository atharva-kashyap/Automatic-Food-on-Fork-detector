import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Apply the mask on the image outputted from t-test
# Separately, try Grab Cut on the image

# And then add some binary thresholding on the T-test image and then get apply the mask on that

if "__main__" == __name__:
    folderToRead = "//wsl.localhost/Ubuntu-20.04/home/atharvak/prl/forktipFoodDetection_ws/src/images/"
    maskToRead = "Masks/MaskWithCarrot_2_7_23.png"
    imgToRead = "grabCutAfterTtest/p_thresholded.png"
    
    image = cv.imread(folderToRead + imgToRead)
    assert image is not None, "file could not be read, check with os.path.exists()"
    mask = np.zeros(image.shape[:2], np.uint8)
    newmask = cv.imread(folderToRead + maskToRead, cv.IMREAD_GRAYSCALE)
    assert newmask is not None, "file could not be read, check with os.path.exists()"

    cv.imshow("Mask", mask)
    cv.imshow("NewMask", newmask)
    cv.imshow("Image", image)
    cv.waitKey(0)

    # any values in the mask greater than 0 should be background
    # any values equal to 0 (black) should be "probable foreground"
    mask[newmask > 0] = cv.GC_PR_BGD
    mask[newmask == 0] = cv.GC_FGD

    # internally used arrays (tbh, i don't know what these are doing)
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    mask, bgdModel, fgdModel = cv.grabCut(image, mask, None, bgModel, fgModel, 5, cv.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = image * mask[:, :, np.newaxis]
    cv.imshow("AfterImage", img)
    cv.imshow("BeforeImage", image)
    cv.waitKey(0)
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()