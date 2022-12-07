import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import rosbag
import rospy

if __name__ == "__main__":
    IN_DIR = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/"
    # OUT_DIR = "/home/amalnanavati/workspaces/amal_noetic_ws/src/ada_amal/automated_bite_detection/"

    rosbagname = "ada_camera_all_images_compressed_bagfile"

    depth_topic = '/camera/aligned_depth_to_color/image_raw'
    camera_topic = '/camera/color/image_raw'

    bag = rosbag.Bag(os.path.join(IN_DIR, rosbagname+".bag"))
    bridge = CvBridge()
    latest_camera_img = None
    depth_image_i = -1
    for topic, msg, t in bag.read_messages(topics = [depth_topic, camera_topic]):
        if topic == depth_topic:
            depth_image_i += 1
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            print(depth_image_i, img[479, 639], img[0, 639])
            img = cv2.convertScaleAbs(img)
            # img[470:480,620:640] = 255
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            if latest_camera_img is not None:
                cv2.imshow("depth_image", np.concatenate((img, latest_camera_img), axis=1))
                if depth_image_i in [0, 1, 11, 18, 21, 22, 23, 33, 34, 35, 36]:
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(500)
        elif topic == camera_topic:
            latest_camera_img = cv2.cvtColor(bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'), cv2.COLOR_RGB2BGR)

        # cv2.imwrite(os.path.join(OUT_DIR, "%s/%d.png" % (rosbagname, int(msg.header.stamp.to_sec()*10**9))), imageToSave)
        if rospy.is_shutdown(): break
    bag.close()
    pass
