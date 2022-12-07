import cv2
import os

if __name__ == "__main__":
    FPS = 30
    # IN_DIR = "/Users/amaln/Documents/PRL/ada_amal/automated_bite_detection/2022_11_01_ada_picks_up_carrots_camera_ft_tf"
    IN_DIR = "/Users/amaln/Documents/PRL/ada_amal/automated_bite_detection/2022_11_01_ada_picks_up_carrots_camera_compressed_ft_tf"
    vid_name = os.path.basename(IN_DIR)
    print(vid_name)

    # Get the timestamp of all images
    image_ts = []
    for filename in os.listdir(IN_DIR):
        if os.path.isfile(os.path.join(IN_DIR, filename)):
            try:
                ts = int(filename[:filename.find(".")])
                image_ts.append(ts)
            except Exception as e:
                print(e)
    image_ts.sort()
    start_time = image_ts[0]
    end_time = image_ts[-1]
    # raise Exception((end_time-start_time)/10**9/60)

    # Open the video file
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(os.path.join(IN_DIR, '../%s.mp4' % vid_name), fourcc, FPS, (640, 480), isColor=True)

    i = 0
    current_frame_time = start_time
    while current_frame_time < end_time:
        print("%d/%d" % (i, len(image_ts)))
        # Write the current frame
        img_path = os.path.join(IN_DIR, "%d.png" % image_ts[i])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        out.write(img)

        # Increment the current_frame_time by one frame
        current_frame_time += 10**9/FPS
        while i < len(image_ts)-1 and image_ts[i+1] < current_frame_time:
            i += 1
    # Write the last frame
    img_path = os.path.join(IN_DIR, "%d.png" % image_ts[i])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    out.write(img)

    # Close the video file
    out.release()
