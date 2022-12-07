#!/usr/bin/env python
import rospy
import tf
import tf.transformations

if __name__ == '__main__':
    rospy.init_node('rotate_ft_sensor')

    base_frame = "j2n6s200_link_base"
    ft_sensor_frame = "j2n6s200_forque"

    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()

        try:
            (trans, rot) = listener.lookupTransform(ft_sensor_frame, base_frame, rospy.Time(0))
            br.sendTransform((0, 0, 0),
                # rot,
                tf.transformations.quaternion_from_matrix(tf.transformations.inverse_matrix(tf.transformations.quaternion_matrix(rot))),
                rospy.Time.now(),
                "j2n6s200_forque_rotated",
                ft_sensor_frame)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print(e)
            continue
