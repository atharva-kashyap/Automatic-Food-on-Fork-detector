#!/usr/bin/env python3

# if __name__ == "__main__":
#    print(" hello ")

import rosbag
import rospy
from cv_bridge import CvBridge
import cv2
import os
import os.path

if __name__ == "__main__":
    rospy.init_node('convertToCv2', anonymous=False)
    # "2022_11_29_mostly_grape_failures_some_successes"
    # "ada_camera_all_images_compressed_bagfile"
    rosbagname = "2022_11_29_2_grape_acquisition_success_1_failure"
    prependLocation = "/home/atharvak/prl/forktipFoodDetection_ws/src"

    # is there a place to store the images?
    imageStoreDir = prependLocation + "/images/"
    os.chdir(imageStoreDir)
    folderName = rosbagname + "_Images"
    if not os.path.isdir(folderName):
        os.mkdir(folderName)

    # initialize a bag file
    bag = rosbag.Bag(prependLocation + "/rosbags/" + rosbagname + '.bag')
    bridge = CvBridge()

    # '/camera/aligned_depth_to_color/image_raw'

    for topic, msg, t in bag.read_messages(topics = ['/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw']):
        print("header", msg.header)
        if topic == "/camera/color/image_raw":
            imageToSave = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            imageToSave = cv2.cvtColor(imageToSave, cv2.COLOR_RGB2BGR)
            pathToSave = imageStoreDir + folderName + "/"
            os.chdir(pathToSave)
            print(os.listdir(pathToSave))
            pathToSave2 = str((msg.header.stamp.secs * 1000000000) + msg.header.stamp.nsecs) + ".png"
            cv2.imwrite(pathToSave2, imageToSave)
        # else:
        #     imageToSave = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        #     pathToSave = imageStoreDir + rosbagname + "/"
        #     os.chdir(pathToSave)
        #     pathToSave2 = str((msg.header.stamp.secs * 1000000000) + msg.header.stamp.nsecs) + "_depth.png"
        #     cv2.imwrite(pathToSave2, imageToSave)
        if rospy.is_shutdown(): break
    bag.close()
    print("done")
