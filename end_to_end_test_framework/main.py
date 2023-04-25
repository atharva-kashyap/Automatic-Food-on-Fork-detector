import cv2
from food_on_fork_detectors.FoodOnForkInterface import FoodOnForkInterface
from food_on_fork_detectors.DummyClassifier import DummyClassifier
from food_on_fork_detectors.RandomClassifier import RandomClassifier
from food_on_fork_detectors.CVImageDiffTTestClassifier import CVImageDiffTTestClassifier
import numpy as np
import os
from pathlib import Path
import pprint
from rosbags.highlevel import AnyReader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from typing import List, Dict, Tuple
import yaml


def get_all_rosbags_and_success(
    base_dir: str,
    rosbag_name: str = "move_out_image_ft.bag",
    status_name: str = "result.yaml",
) -> List[Tuple[str, bool]]:
    """
    Takes in a base directory and gets the paths for all rosbags with name
    {base_dir}/*/{rosbag_name} and that have a sister file {base_dir}/*/{status_name}

    Returns a list of tuples (rosbag_path, status)
    """
    retval = []
    for root, dirs, files in os.walk(base_dir):
        if rosbag_name in files and status_name in files:
            # Get status
            with open(os.path.join(root, status_name), "r") as f:
                success = yaml.safe_load(f)["success"]
            retval.append((os.path.join(root, rosbag_name), success))
    return retval


def analyze_results(
    preds_per_algorithm: Dict[str, List[bool]], actual: List[bool]
) -> None:
    """
    Prints the results and various metrics about them
    """
    for algorithm_name in algorithms.keys():
        print("Algorithm: %s" % algorithm_name)
        print(
            "\tAccuracy: %f"
            % accuracy_score(actual, preds_per_algorithm[algorithm_name])
        )
        print(
            "\tPrecision: %f"
            % precision_score(actual, preds_per_algorithm[algorithm_name])
        )
        print(
            "\tRecall: %f" % recall_score(actual, preds_per_algorithm[algorithm_name])
        )
        print("\tF1 Score: %f" % f1_score(actual, preds_per_algorithm[algorithm_name]))


if __name__ == "__main__":
    # Parameters -- change these as necessary
    viz: bool = True  # Whether to visualize the rosbag images
    # # Running on Amal's local machine
    # rss_experiments_folder_path: str = "/Users/amalnanavati/Documents/PRL/forktip_food_detection_cv/2023_RSS_experiments/"
    # Running on weebo
    # rss_experiments_folder_path: str = (
    #     "/home/ekgordon/Workspace/ada_ws/src/ada_feeding/data/"
    # )

    # Atharva's Rosbags location path
    rss_experiments_folder_path: str = (
        "/home/atharvak/prl/forktipFoodDetection_ws/src/ethan_rosbags"
    )

    # Get all the rosbags in rss_experiments_folder_path along with their
    # success status
    rosbags_and_success: List[Tuple[str, bool]] = get_all_rosbags_and_success(
        rss_experiments_folder_path
    )

    # The list of algorithms to test
    seed: int = int(time.time() * 10**9)  # Seed randomness with the current time (ns)
    # NOTE: For reproducibility, include the seed in the algorithm name
    algorithms: Dict[str, FoodOnForkInterface] = {
        # True is the majority class, occuring in 0.5625 of the data
        # Hence, to be better than baseline, our classifier needs to have an
        # accuracy of > 0.5625 (and an F1 score of >= 0.72)
        # "MajorityClass": DummyClassifier(pred=True),
        # # In expectation, RandomClassifier should have an accuracy of 0.50
        # "RandomClassifier(seed=%d)" % seed: RandomClassifier(seed=seed),
        # CVImageDiffT-TestClassifier
        "CVImageDiffTTestClassifier": CVImageDiffTTestClassifier(cv2.COLOR_RGB2HSV),
    }

    # Values to store predictions versus actual values
    preds_per_algorithm: Dict[str, List[bool]] = {
        algorithm_name: [] for algorithm_name in algorithms.keys()
    }
    actual: List[bool] = []

    # Iterate over the rosbags
    for i in range(len(rosbags_and_success)):
        rosbag, success = rosbags_and_success[i]
        print("%d/%d: %s, %s" % (i, len(rosbags_and_success), rosbag, success))
        with AnyReader([Path(rosbag)]) as reader:
            connections = [
                x
                for x in reader.connections
                if x.topic == "/camera/color/image_raw/compressed"
            ]

            # Initialize the algorithms
            for algorithm in algorithms.values():
                algorithm.initialize()

            # For every image in the rosbag, send it to the algorithms
            for connection, timestamp, rawdata in reader.messages(
                connections=connections
            ):
                # Get the image data
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

                # Send it to every algorithm
                for algorithm in algorithms.values():
                    algorithm.next_image(img)

                # Optional: Visualize it
                if viz:
                    cv2.imshow("img", img)
                    cv2.waitKey(50)

            # Get the predictions
            for algorithm_name, algorithm in algorithms.items():
                preds_per_algorithm[algorithm_name].append(algorithm.predict())
            # Get the actual value
            actual.append(success)

    # Print and visualize the results
    analyze_results(preds_per_algorithm, actual)
