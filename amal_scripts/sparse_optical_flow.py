import cv2 as cv
import math
import numpy as np
import os

if __name__ == "__main__":
    # images_dir = "/Users/amaln/Documents/PRL/ada_amal/automated_bite_detection/ada_camera_all_images_compressed_bagfile/"
    # start_time = 1658882626566284715 # 1658882604054431928 # 0 # 1658882579329679616 # 1658882567200893896 #
    # p0 = np.array([
    #     [[390, 230]],
    #     [[400, 230]],
    #     [[410, 230]],
    #     [[420, 230]],
    #     [[390, 210]],
    #     [[390, 200]],
    #     [[390, 190]],
    #     [[390, 180]],
    # ], dtype=np.float32)
    # images_dir = "/Users/amaln/Documents/PRL/ada_amal/automated_bite_detection/2022_11_01_ada_picks_up_carrots_camera_ft_tf/"
    # start_time = 1667340055153559296
    # p0 = np.array([
    #     [[424, 230]],
    #     [[435, 230]],
    #     [[446, 230]],
    #     [[457, 230]],
    #     [[440, 220]],
    #     [[440, 210]],
    #     [[440, 200]],
    #     [[440, 190]],
    # ], dtype=np.float32)
    images_dir = "/Users/amaln/Documents/PRL/ada_amal/automated_bite_detection/2022_11_01_ada_picks_up_carrots_camera_compressed_ft_tf/"
    start_time = 1667339196734672896 # 1667339206378519040 # 
    p0 = np.array([
        [[423, 230]],
        [[434, 230]],
        [[445, 230]],
        [[456, 230]],
        [[440, 220]],
        [[440, 210]],
        [[440, 200]],
        [[440, 190]],
    ], dtype=np.float32)
    end_time = math.inf

    filepaths = {}
    for filename in os.listdir(images_dir):
        filepath = os.path.join(images_dir, filename)
        if os.path.isfile(filepath) and "png" in filename.lower():
            timestamp = int(filename[:-4])
            filepaths[timestamp] = filepath

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    prev_img = None
    next_img = None
    for timestamp in sorted(filepaths.keys()):
        if timestamp < start_time: continue
        if timestamp > end_time: break
        print(timestamp)
        filepath = filepaths[timestamp]
        prev_img = next_img
        # frame = cv.imread(filepath, cv.IMREAD_COLOR)
        frame = cv.cvtColor(cv.imread(filepath, cv.IMREAD_COLOR), cv.COLOR_RGB2BGR)
        next_img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        if prev_img is None:
            # p0 = cv.goodFeaturesToTrack(next_img, mask = None, **feature_params)
            # raise Exception(p0, p0.shape)
            mask = np.zeros_like(frame)
            continue
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_img, next_img, p0, None, **lk_params)
        # p1 = p0
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 1, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        p0 = good_new.reshape(-1, 1, 2)
    cv.destroyAllWindows()
