import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
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

food_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cropped_images/depth_img/food/")
no_food_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cropped_images/depth_img/no_food/")

X = []
y = []
rosbag_names = []

min_dist = 310
max_dist = 370

# Bin 0 will always be "outside the frustrum."
# Bins 1 to (num_bins-1) will be the bins for the range of distances.
num_bins = 3
bin_size = (max_dist - min_dist) / (num_bins-1)

def get_bin(dist):
    if dist < min_dist: return 0
    if dist > max_dist: return 0
    if dist == max_dist: return num_bins-1
    return int((dist - min_dist) // bin_size) + 1

# Read all the PNG files in the food directory
for filename in os.listdir(food_dir):
    if filename.endswith(".png"):
        rosbag_name = filename.split("_")[0]

        filepath = os.path.join(food_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        shape = img.shape

        img_binned = np.vectorize(get_bin)(img).astype('uint8')

        # plt_show_depth_img(img_binned)

        X.append(img_binned.flatten())
        y.append(1)
        rosbag_names.append(rosbag_name)

# Read all the PNG files in the no food directory
for filename in os.listdir(no_food_dir):
    if filename.endswith(".png"):
        rosbag_name = filename.split("_")[0]

        filepath = os.path.join(no_food_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        img_binned = np.vectorize(get_bin)(img).astype('uint8')

        # plt_show_depth_img(img_binned)

        X.append(img_binned.flatten())
        y.append(0)
        rosbag_names.append(rosbag_name)

X = np.array(X)
y = np.array(y)
rosbag_names = np.array(rosbag_names)
print("X", X.shape)
print("y", y.shape)
print("rosbag_names", rosbag_names.shape, np.unique(rosbag_names, return_counts=True))

# Split the train and test set
seed = int(time.time())#1692210087#
print("seed", seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Train a Naive Bayes classifier on the data
if num_bins == 2:
    clf = BernoulliNB()
else:
    clf = CategoricalNB(min_categories=num_bins)
clf.fit(X_train, y_train)

# Get the predictions on the train set
print("X_train", X_train.shape)
y_pred_train = clf.predict(X_train)
y_pred_train_prob = clf.predict_proba(X_train)
print("y_pred_train_prob", y_pred_train_prob, y_pred_train_prob.min(), y_pred_train_prob.max(), y_pred_train_prob.mean())

# Get the predictions on the test set
print("X_test", X_test.shape)
y_pred_test = clf.predict(X_test)
y_pred_test_prob = clf.predict_proba(X_test)
print("y_pred_test_prob", y_pred_test_prob, y_pred_test_prob.min(), y_pred_test_prob.max(), y_pred_test_prob.mean())

# Get the train accuracy
acc_train = accuracy_score(y_train, y_pred_train)
print("Train accuracy", acc_train)
print("Train confusion matrix\n", confusion_matrix(y_train, y_pred_train))

# Get the test accuracy
acc_test = accuracy_score(y_test, y_pred_test)
print("Test accuracy", acc_test)
print("Test confusion matrix\n", confusion_matrix(y_test, y_pred_test))

# # Visualize what was learnt
# print("clf.feature_log_prob_", [m.shape for m in clf.feature_log_prob_])
# print("clf.feature_log_prob_", [np.sum(np.e**m, axis=1) for m in clf.feature_log_prob_])
# print("clf.feature_log_prob_", clf.feature_log_prob_, len(clf.feature_log_prob_), clf.feature_log_prob_[-1].shape)
# print("clf.class_count_", clf.class_count_, len(clf.class_count_), clf.class_count_[0].shape)

# CategoricalNB
if clf.__class__ == CategoricalNB:
    conditional_probabilities = [np.e**m for m in clf.feature_log_prob_]
    prob_that_pixel_is_in_range_given_fof = []
    prob_that_pixel_is_in_range_given_no_fof = []
    for i in range(len(conditional_probabilities)):
        prob_that_pixel_is_in_range_given_fof.append(1.0-conditional_probabilities[i][1,0])
        prob_that_pixel_is_in_range_given_no_fof.append(1.0-conditional_probabilities[i][0,0])
# BernoulliNB
else:
    prob_that_pixel_is_in_range_given_fof = np.e**clf.feature_log_prob_[1,:]
    prob_that_pixel_is_in_range_given_no_fof = np.e**clf.feature_log_prob_[0,:]

# Show graphs
prob_that_pixel_is_in_range_given_fof = np.array(prob_that_pixel_is_in_range_given_fof).reshape(shape)
prob_that_pixel_is_in_range_given_no_fof = np.array(prob_that_pixel_is_in_range_given_no_fof).reshape(shape)
plt_show_depth_img(prob_that_pixel_is_in_range_given_no_fof, title="Probability that a pixel is in range given no food on the fork")
plt_show_depth_img(prob_that_pixel_is_in_range_given_fof, title="Probability that a pixel is in range given food on the fork")
plt_show_depth_img(prob_that_pixel_is_in_range_given_fof-prob_that_pixel_is_in_range_given_no_fof, title="P(px in range | FoF) - P(px in range | no FoF)")
