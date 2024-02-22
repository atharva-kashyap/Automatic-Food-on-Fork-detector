import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
import time

def normalize_to_uint8(img):
    # Normalize the image to 0-255
    img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
    return img_normalized

def plt_show_depth_img(img, show=True, title=""):
    # Plot the depth image
    plt.imshow(img)
    plt.title(title)
    plt.colorbar()
    if show: plt.show()

def cv_show_normalized_depth_img(img, wait=True):
    # Show the normalized depth image
    img_normalized = normalize_to_uint8(img)
    cv2.imshow("img", img_normalized)
    if wait: cv2.waitKey(0)

# NOTE: These images are **already cropped**!
food_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cropped_images/depth_img/food/")
no_food_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cropped_images/depth_img/no_food/")

# I'm assuming these are the hardcoded coordinates of the crop, based on 
# https://github.com/personalrobotics/ada_feeding/blob/7edbf3f0a822fe74687248501bd18184f07e6862/ada_feeding_perception/ada_feeding_perception/Crop_Images.py#L14C27-L14C93
# The dimensions match.
crop_x_min=297
crop_y_min=248
crop_x_max=425
crop_y_max=332

# Hardcoded for our RealSense from https://github.com/personalrobotics/feeding_web_interface/blob/8c5f2f905b5aa0621bae93fb5f8d81559e4b6281/feeding_web_app_ros2_test/feeding_web_app_ros2_test/DummyRealSense.py#L102
camera_matrix_k = [
    614.5933227539062,
    0.0,
    312.1358947753906,
    0.0,
    614.6914672851562,
    223.70831298828125,
    0.0,
    0.0,
    1.0,
]

no_food_points = []
no_food_rosbag_names = []
food_points = []
food_rosbag_names = []

min_dist = 310
max_dist = 370

def deproject_depth_image(depth_img):
    # For every non-zero depth point, calculate the corresponding 3D point
    # in the camera frame
    us = np.tile(np.arange(depth_img.shape[1]), (depth_img.shape[0], 1)) + crop_x_min
    # print("us", us, us.shape, us.min(), us.max())
    vs = np.tile(np.arange(depth_img.shape[0]), (depth_img.shape[1], 1)).T + crop_y_min
    # print("vs", vs, vs.shape, vs.min(), vs.max())

    # Mask out any points where depth is 0
    mask = depth_img > 0
    depth_img_masked = depth_img[mask]
    us_masked = us[mask]
    vs_masked = vs[mask]
    # print("depth_img_masked", depth_img_masked.shape, "us_masked", us_masked.shape, "vs_masked", vs_masked.shape)

    xs = np.multiply(us_masked - camera_matrix_k[2], np.divide(depth_img_masked, 1000.0 * camera_matrix_k[0]))
    ys = np.multiply(vs_masked - camera_matrix_k[5], np.divide(depth_img_masked, 1000.0 * camera_matrix_k[4]))
    zs = np.divide(depth_img_masked, 1000.0)

    img_3d_points = np.concatenate((xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]), axis=1)

    return np.array(img_3d_points)

# Read all the PNG files in the no food directory
no_food_len = len([filename for filename in os.listdir(no_food_dir) if filename.endswith(".png")])
i = 0
for filename in os.listdir(no_food_dir):
    if filename.endswith(".png"):
        i += 1
        # print(f"Processing no_food {i}/{no_food_len}")

        rosbag_name = filename.split("_")[0]

        filepath = os.path.join(no_food_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        # 84 x 128 images, pre-cropped

        # Remove all points not within the frustrum
        img_masked = np.where(img < min_dist, 0, img)
        img_masked = np.where(img_masked > max_dist, 0, img)

        # For every non-zero depth point, calculate the corresponding 3D point
        # in the camera frame
        img_3d_points = deproject_depth_image(img_masked)

        # plt_show_depth_img(img)#_masked)

        no_food_points.append(img_3d_points)
        no_food_rosbag_names.append(rosbag_name)

num_no_food_points_per_img = np.array([point_cloud.shape[0] for point_cloud in no_food_points])
print("num_no_food_points_per_img", num_no_food_points_per_img.min(), num_no_food_points_per_img.max(), num_no_food_points_per_img.mean(), num_no_food_points_per_img.std(), np.percentile(num_no_food_points_per_img, 25), np.percentile(num_no_food_points_per_img, 50), np.percentile(num_no_food_points_per_img, 75))
print("num_no_food_points_per_img", np.sum(num_no_food_points_per_img < 40))

# Read all the PNG files in the food directory
food_len = len([filename for filename in os.listdir(food_dir) if filename.endswith(".png")])
i = 0
for filename in os.listdir(food_dir):
    if filename.endswith(".png"):
        i += 1
        # print(f"Processing food {i}/{food_len}")

        rosbag_name = filename.split("_")[0]

        filepath = os.path.join(food_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        shape = img.shape

        # Remove all points not within the frustrum
        img_masked = np.where(img < min_dist, 0, img)
        img_masked = np.where(img_masked > max_dist, 0, img)

        # For every non-zero depth point, calculate the corresponding 3D point
        # in the camera frame
        img_3d_points = deproject_depth_image(img_masked)

        # plt_show_depth_img(img)#_masked)

        food_points.append(img_3d_points)
        food_rosbag_names.append(rosbag_name)

num_food_points_per_img = np.array([point_cloud.shape[0] for point_cloud in food_points])
print("num_food_points_per_img", num_food_points_per_img.min(), num_food_points_per_img.max(), num_food_points_per_img.mean(), num_food_points_per_img.std(), np.percentile(num_food_points_per_img, 25), np.percentile(num_food_points_per_img, 50), np.percentile(num_food_points_per_img, 75))
print("num_food_points_per_img", np.sum(num_food_points_per_img < 40))
# raise Exception()

# Split the no_food_points into train and test set
seed = int(time.time())#1692210087#
print("seed", seed)
no_food_points_train, no_food_points_test = train_test_split(no_food_points, train_size=500, random_state=seed)
print("Training Set Size", len(no_food_points_train))

# Shuffle food_points
np.random.seed(seed)
np.random.shuffle(food_points)

# # Flatten no_food_points_train to a single population
# no_food_points_train = np.concatenate(no_food_points_train, axis=0)
# print("no_food_points_train", no_food_points_train.shape)

# # Compute the mean and sample covariance of the no_food_points_train
# no_food_points_train_mean = np.mean(no_food_points_train, axis=0)
# no_food_points_train_cov = np.cov(no_food_points_train, rowvar=False, bias=False)
# print("no_food_points_train_mean", no_food_points_train_mean, no_food_points_train_mean.shape)
# print("no_food_points_train_cov", no_food_points_train_cov, no_food_points_train_cov.shape)

# Compute the means and sample covariances for each depth image in no_food_points_train and no_food_points_test
no_food_points_train_means = np.array([np.mean(point_cloud, axis=0) for point_cloud in no_food_points_train])
no_food_points_train_covs = np.array([np.cov(point_cloud, rowvar=False, bias=False) for point_cloud in no_food_points_train])
no_food_points_train_ns = np.array([point_cloud.shape[0] for point_cloud in no_food_points_train])

def hotellings_test(pop1_mean, pop1_cov, n1, pop2):
    """
    Run the Hotelling two-sample test where populations have unequal covariances.

    Based on:
      1. https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Hotellings_Two-Sample_T2.pdf
    """
    pop2_mean = np.mean(pop2, axis=0)
    pop2_cov = np.cov(pop2, rowvar=False, bias=False)
    n2 = pop2.shape[0]

    # Calculate the S matrix
    S = (pop1_cov / n1) + (pop2_cov / n2)

    # Calculate the T^2 statistic
    t_sq = np.dot(np.dot((pop1_mean - pop2_mean).T, np.linalg.inv(S)), (pop1_mean - pop2_mean))

    # Calculate the degrees of freedom
    df1 = pop1_mean.shape[0]
    df2 = (
        np.trace(np.dot(S, S)) + np.trace(S)**2.0
    ) / (
        (
            (np.trace(np.dot(pop1_cov / n1, pop1_cov / n1)) + np.trace(pop1_cov / n1)**2.0) / (n1 - 1)
        ) + (
            (np.trace(np.dot(pop2_cov / n2, pop2_cov / n2)) + np.trace(pop2_cov / n2)**2.0) / (n2 - 1)
        )
    )

    # Calculate the corresponding F value
    F_vals = (df2 - df1 + 1) / (df1 * df2) * t_sq

    # Calculate the p value
    p = 1-scipy.stats.f.cdf(F_vals, df1, df2 - df1 + 1)

    # print("pop1_mean", pop1_mean, "pop1_cov", pop1_cov, "n1", n1)
    # print("pop2_mean", pop2_mean, "pop2_cov", pop2_cov, "n2", n2)
    # print("S", S, "t_sq", t_sq, "df1", df1, "df2", df2, "F_vals", F_vals, "p", p)
    # raise Exception("Stop here")

    return p

def compute_p_value_bounds(point_cloud):
    """
    For every distribution in no_food_points_train, run the Hotelling two-sample t-sq test
    to see if the distribution of point_cloud is significantly different.
    """
    ps = []
    for i in range(len(no_food_points_train)):
        # print(f"{i}/{len(no_food_points_train)}")
        no_food_points_train_mean = no_food_points_train_means[i]
        no_food_points_train_cov = no_food_points_train_covs[i]
        no_food_points_train_n = no_food_points_train_ns[i]
        p = hotellings_test(
            no_food_points_train_mean, 
            no_food_points_train_cov, 
            no_food_points_train_n, 
            point_cloud
        )
        ps.append(p)
    ps = np.array(ps)
    lower_bound = ps.max()
    upper_bound = min(ps.sum(), 1.0)
    return lower_bound, upper_bound

# TODO: Add a min threshold of points that must be in the image, where if those number of points are
# not there, the classifier returns unable to classify.

# For every set of points (every depth image) in no_food_points_test, run the Hotelling two-sample t-sq test
# with unequal covariances
num_cap = math.inf # 60 # 

y_true = []
y_pred = []
NO_FOOD_LABEL = 0
FOOD_LABEL = 1
UNSURE_LABEL = 2
NO_FOOD_THRESH = 0.75
FOOD_THRESH = 0.05

times = []
no_food_ps = []
no_food_len = len(no_food_points_test)
i = 0
for point_cloud in no_food_points_test:
    i += 1
    print(f"Processing no_food_test {i}/{no_food_len}", end="")
    start_time = time.time()
    p_lower, p_upper = compute_p_value_bounds(point_cloud)
    times.append(time.time()-start_time)
    print(f" p={(p_lower, p_upper)}, time={times[-1]}")
    no_food_ps.append((p_lower, p_upper))
    if i > num_cap: break
no_food_ps = np.array(no_food_ps)

y_true.extend([NO_FOOD_LABEL]*no_food_ps.shape[0])
y_pred.extend(np.where(no_food_ps[:,0] > NO_FOOD_THRESH, NO_FOOD_LABEL, np.where(no_food_ps[:,1] < FOOD_THRESH, FOOD_LABEL, UNSURE_LABEL)))

tn = np.sum(no_food_ps[:,0] > NO_FOOD_THRESH)
fp = len(no_food_ps) - tn
print("no_food", tn, fp, float(tn) / (tn + fp))

food_ps = []
food_len = len(food_points)
i = 0
for point_cloud in food_points:
    i += 1
    print(f"Processing food {i}/{food_len}", end="")
    start_time = time.time()
    p_lower, p_upper = compute_p_value_bounds(point_cloud)
    times.append(time.time()-start_time)
    print(f" p={(p_lower, p_upper)}, time={times[-1]}")
    food_ps.append((p_lower, p_upper))
    if i > num_cap: break
food_ps = np.array(food_ps)

y_true.extend([FOOD_LABEL]*food_ps.shape[0])
y_pred.extend(np.where(food_ps[:,0] > NO_FOOD_THRESH, NO_FOOD_LABEL, np.where(food_ps[:,1] < FOOD_THRESH, FOOD_LABEL, UNSURE_LABEL)))

tp = np.sum(food_ps[:,1] < FOOD_THRESH)
fn = len(food_ps) - tp
print("food", tp, fn, float(tp) / (tp + fn))

# Print the confusion matrix
print("Confusion matrix")
print(confusion_matrix(y_true, y_pred))

print("Time", np.mean(times), np.std(times), np.percentile(times, 25), np.percentile(times, 50), np.percentile(times, 75))

# Save CSVs of the p values
np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out/no_food_ps.txt"), no_food_ps, delimiter=",")
np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out/food_ps.txt"), food_ps, delimiter=",")