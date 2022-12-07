import matplotlib.pyplot as plt
import numpy as np
import rosbag
import rospy
from scipy.ndimage import gaussian_filter1d
import tf.transformations
import tf2_py

def transform_to_matrix(transform_msg):
    m = tf.transformations.quaternion_matrix([
        transform_msg.rotation.x,
        transform_msg.rotation.y,
        transform_msg.rotation.z,
        transform_msg.rotation.w,
    ])
    m[0][3] = transform_msg.translation.x
    m[1][3] = transform_msg.translation.y
    m[2][3] = transform_msg.translation.z
    return m

if __name__ == "__main__":
    # This rosbag should contain the tf topics and the F/T sensor topics, at
    # a minimum
    # rosbag_filepath = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/2022_11_01_ada_picks_up_carrots_camera_ft_tf.bag"
    # food_on_fork_time_periods = [
    #     (1667340055.720348, 1667340074.801857),
    #     (1667340101.459190, 1667340121.605187),
    #     (1667340147.609328, 1667340164.000023),
    # ]
    rosbag_filepath = "/home/amalnanavati/workspaces/amal_noetic_ws/rosbags/2022_11_01_ada_picks_up_carrots_camera_compressed_ft_tf.bag"
    food_on_fork_time_periods = [
        (1667339148.504503, 1667339167.741674),
        (1667339206.004664, 1667339224.967138),
        (1667339254.752515, 1667339277.261415),
    ]

    # Get the topics
    tf_static_topic = "/tf_static"
    tf_topic = "/tf"
    force_topic = "/forque/forqueSensor"
    topics = [
        tf_static_topic,
        tf_topic,
        force_topic,
    ]
    base_frame = "j2n6s200_link_base"
    ft_sensor_frame = "j2n6s200_forque"

    # Open the bag
    bag = rosbag.Bag(rosbag_filepath)
    start_time = bag.get_start_time()
    end_time = bag.get_end_time()

    # Open a tf buffer
    tf_buffer = tf2_py.BufferCore(rospy.Duration((end_time-start_time)*2.0)) # times 2 is to give it enough space in case some messages were backdated

    # Iterate over the msgs in the rosbag
    force_data_raw = []
    for topic, msg, timestamp in bag.read_messages(topics=topics):
        if topic == tf_static_topic:
            for transform in msg.transforms:
                tf_buffer.set_transform_static(transform, "default_authority")
        elif topic == tf_topic:
            for transform in msg.transforms:
                tf_buffer.set_transform(transform, "default_authority")
        elif topic == force_topic:
            force_data_raw.append(msg)

    # **Rotate** every force message to be in the world frame, so we can
    # determine the force in the direction of gravity
    # force_data_static_frame = []
    ts, xs_raw, ys_raw, zs_raw, xs_rot, ys_rot, zs_rot = [], [], [], [], [], [], []
    for msg in force_data_raw:
        # TODO: Determine whether the transform should go in this direction
        # or the opposite.
        try:
            # ft_transform times a vector in base_frame yields a vector in ft_sensor_frame
            ft_transform = tf_buffer.lookup_transform_core(ft_sensor_frame, base_frame, msg.header.stamp)
            ft_transform_matrix = transform_to_matrix(ft_transform.transform)
            ft_rotation_matrix = ft_transform_matrix[0:3, 0:3]

            force_vec_raw = np.array([[msg.wrench.force.x], [msg.wrench.force.y], [msg.wrench.force.z]])
            force_vec_rotated = np.dot(np.linalg.inv(ft_rotation_matrix), force_vec_raw)

            ts.append(msg.header.stamp.to_sec())
            xs_raw.append(force_vec_raw[0,0])
            ys_raw.append(force_vec_raw[1,0])
            zs_raw.append(force_vec_raw[2,0])
            xs_rot.append(force_vec_rotated[0,0])
            ys_rot.append(force_vec_rotated[1,0])
            zs_rot.append(force_vec_rotated[2,0])
        except Exception as e:
            print(e)
            continue

    # Graph the rotated force and torque
    config = [
        [(xs_raw, "X (local)"),
        (ys_raw, "Y (local)"),
        (zs_raw, "Z (local)")],
        [(xs_rot, "X (world)"),
        (ys_rot, "Y (world)"),
        (zs_rot, "Z (world)")]
    ]
    fig, axes = plt.subplots(nrows=len(config), ncols=len(config[0]), sharex='all', sharey='row', figsize=(16,8))
    for i in range(len(config)):
        for j in range(len(config[i])):
            axes[i,j].plot(ts, config[i][j][0], 'b-', label="force")
            axes[i,j].plot(ts, gaussian_filter1d(config[i][j][0], 3), 'k-', label="force smoothened")
            # axes[j].grid('both')
            axes[i,j].xaxis.set_tick_params(labelbottom=True)
            axes[i,j].yaxis.set_tick_params(labelbottom=True)
            axes[i,j].set_ylabel('Force %s' % config[i][j][1])
            axes[i,j].set_xlabel('Time (secs)')
            for start, end in food_on_fork_time_periods:
                axes[i,j].axvline(x=start, c="r", label="food-on-fork START")
                axes[i,j].axvline(x=end, c="c", label="food-on-fork END")
            # axes[i,j].legend()
            axes[i,j].axhline(y=0, c='k')

    plt.show()
