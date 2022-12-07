import cv2 as cv
import math
import numpy as np
import os

if __name__ == "__main__":
    images_dir = "/Users/amaln/Documents/PRL/ada_amal/automated_bite_detection/2022_11_01_ada_picks_up_carrots_camera_ft_tf/"
    start_time = 0 # 1658882604054431928 # 1658882567200893896 #
    end_time = math.inf

    filepaths = {}
    for filename in os.listdir(images_dir):
        filepath = os.path.join(images_dir, filename)
        if os.path.isfile(filepath) and "png" in filename.lower():
            timestamp = int(filename[:-4])
            filepaths[timestamp] = filepath

    prev_img = None
    next_img = None
    flow = None
    # max_mag, min_mag = 30, 0
    max_mag, min_mag = 0, math.inf
    for timestamp in sorted(filepaths.keys()):
        if timestamp < start_time: continue
        if timestamp > end_time: break
        print(timestamp)
        filepath = filepaths[timestamp]
        prev_img = next_img
        frame = cv.imread(filepath, cv.IMREAD_COLOR)
        # frame = cv.cvtColor(cv.imread(filepath, cv.IMREAD_COLOR), cv.COLOR_RGB2BGR)
        next_img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        if prev_img is None:
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            continue

        flow = cv.calcOpticalFlowFarneback(prev_img, next_img, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2

        # # Clip and scale magnitude
        # mag[mag > max_mag] = max_mag
        # mag[mag < min_mag] = min_mag
        # hsv[..., 2] = (mag - min_mag)/(max_mag - min_mag)*255

        # # Normalize based on global max and min
        # max_mag = max(np.max(mag), max_mag)
        # min_mag = min(np.min(mag), min_mag)
        # hsv[..., 2] = (mag - min_mag)/(max_mag - min_mag)*255

        # Normalize per frame
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # # Binary Threshold on Magnitude
        # _, mag = cv.threshold(mag, 2, 255, cv.THRESH_BINARY)
        # hsv[..., 2] = mag

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('next_img', np.concatenate((bgr, frame), axis=1))
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', next_img)
            cv.imwrite('opticalhsv.png', bgr)
    cv.destroyAllWindows()
