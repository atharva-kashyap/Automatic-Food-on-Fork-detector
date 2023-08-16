import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import time

def normalize_to_uint8(img):
    # Normalize the image to 0-255
    img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
    return img_normalized

def plt_show_depth_img(img, show=True):
    # Plot the depth image
    plt.imshow(img)
    plt.colorbar()
    if show: plt.show()

def cv_show_normalized_depth_img(img, wait=True):
    # Show the normalized depth image
    img_normalized = normalize_to_uint8(img)
    cv2.imshow("img", img_normalized)
    if wait: cv2.waitKey(0)

def kld_gauss(u1, s1, u2, s2):
    # From https://jamesmccaffrey.wordpress.com/2021/02/03/the-kullback-leibler-divergence-for-two-gaussian-distributions/
    # general KL two Gaussians
    # u2, s2 often N(0,1)
    # https://stats.stackexchange.com/questions/7440/ +
    # kl-divergence-between-two-univariate-gaussians
    # log(s2/s1) + [( s1^2 + (u1-u2)^2 ) / 2*s2^2] - 0.5
    v1 = s1 * s1
    v2 = s2 * s2
    a = np.log(s2/s1) 
    num = v1 + (u1 - u2)**2
    den = 2 * v2
    b = num / den
    return a + b - 0.5

food_dir = "/Users/amalnanavati/Documents/PRL/forktip_food_detection_cv/cropped_images/depth_img/food/"
no_food_dir = "/Users/amalnanavati/Documents/PRL/forktip_food_detection_cv/cropped_images/depth_img/no_food/"

X = []
y = []

# Read all the PNG files in the food directory
for filename in os.listdir(food_dir):
    if filename.endswith(".png"):
        filepath = os.path.join(food_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        shape = img.shape

        # plt_show_depth_img(img)

        X.append(img.flatten())
        y.append(1)

# Read all the PNG files in the no food directory
for filename in os.listdir(no_food_dir):
    if filename.endswith(".png"):
        filepath = os.path.join(no_food_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        # plt_show_depth_img(img)

        X.append(img.flatten())
        y.append(0)

X = np.array(X)
y = np.array(y)
print("X", X.shape)
print("y", y.shape)

# Split the train and test set
seed = int(time.time())
print("seed", seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

# Train a Naive Bayes classifier on the data
clf = GaussianNB()
clf.fit(X_train, y_train)

# Get the predictions on the train set
y_pred_train = clf.predict(X_train)

# Get the predictions on the test set
y_pred_test = clf.predict(X_test)

# Get the train accuracy
acc_train = accuracy_score(y_train, y_pred_train)
print("Train accuracy", acc_train)
print("Train confusion matrix\n", confusion_matrix(y_train, y_pred_train))

# Get the test accuracy
acc_test = accuracy_score(y_test, y_pred_test)
print("Test accuracy", acc_test)
print("Test confusion matrix\n", confusion_matrix(y_test, y_pred_test))

# Visualize the image of per-pixel KL divergences
means = clf.theta_
plt_show_depth_img(means[0].reshape(shape))
plt_show_depth_img(means[1].reshape(shape))
stds = np.sqrt(clf.var_)
plt_show_depth_img(stds[0].reshape(shape))
plt_show_depth_img(stds[1].reshape(shape))
diff = []
for i in range(means.shape[1]):
    diff.append(kld_gauss(means[1][i], stds[1][i], means[0][i], stds[0][i]))
diff = np.array(diff).reshape(shape)
print("diff", diff)
plt_show_depth_img(diff)


