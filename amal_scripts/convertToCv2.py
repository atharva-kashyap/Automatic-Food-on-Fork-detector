#!/usr/bin/env python
import rosbag
import rospy
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import os.path

if __name__ == "__main__":
    IN_DIR = "/workspace/rosbags/"
    OUT_DIR = "/workspace/src/ada_amal/automated_bite_detection/"

    rospy.init_node('convertToCv2', anonymous=False)

    # rosbagname = "2022_11_01_ada_picks_up_carrots_camera_ft_tf"
    # is_compressed = False
    # rosbagname = "2022_11_01_ada_picks_up_carrots_camera_compressed_ft_tf"
    # is_compressed = True
    # rosbagname = "2022_11_29_mostly_grape_failures_some_successes"
    # is_compressed = False
    # rosbagname = "2022_11_29_3_cantaloupe_successes_6_grape_failures"
    # is_compressed = False
    # rosbagname = "2022_11_29_6_grape_acquisition_successes"
    # is_compressed = False
    # rosbagname = "2022_11_29_3_grape_success_tilted"
    # is_compressed = False
    rosbagname = "2022_11_29_2_grape_acquisition_success_1_failure"
    is_compressed = False

    if not os.path.isdir(os.path.join(OUT_DIR, rosbagname)):
        os.mkdir(os.path.join(OUT_DIR, rosbagname))

    bag = rosbag.Bag(os.path.join(IN_DIR, rosbagname+".bag"))
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages(topics = ['/camera/color/image_raw' + ('/compressed' if is_compressed else '')]):
        if is_compressed:
            imageToSave = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        else:
            imageToSave = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        imageToSave = cv2.cvtColor(imageToSave, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(OUT_DIR, "%s/%d.png" % (rosbagname, int(msg.header.stamp.to_sec()*10**9))), imageToSave)
        if rospy.is_shutdown(): break
    bag.close()
