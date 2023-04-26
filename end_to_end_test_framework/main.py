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
from typing import List, Dict, Tuple, Set
import yaml


def get_all_rosbags_and_success(
    base_dir: str,
    rosbag_name: str = "move_out_image_ft.bag",
    status_name: str = "result.yaml",
    rosbags_to_ignore: Set[str] = set(),
) -> List[Tuple[str, bool]]:
    """
    Takes in a base directory and gets the paths for all rosbags with name
    {base_dir}/*/{rosbag_name} and that have a sister file {base_dir}/*/{status_name}

    Returns a list of tuples (rosbag_path, status)
    """
    retval = []
    for root, dirs, files in os.walk(base_dir):
        # Ignore the rosbags that we know are corrupted
        ignore = False
        for rosbag_to_ignore in rosbags_to_ignore:
            if rosbag_to_ignore in root:
                ignore = True
                break
        if ignore: continue
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

    # These ROS bags were all corrupted, where the stored into, within, and
    # out actions were actually off-by-one from the true action.
    rosbags_to_ignore: Set[str] = set([
        'banana_action0_trial0',
        'banana_action0_trial1',
        'banana_action0_trial2',
        'banana_action0_trial3',
        'banana_action0_trial4',
        'banana_action0_trial5',
        'banana_action0_trial6',
        'banana_action0_trial7',
        'banana_action0_trial8',
        'banana_action0_trial9',
        'banana_action1_trial0',
        'banana_action1_trial1',
        'banana_action1_trial2',
        'banana_action1_trial3',
        'banana_action1_trial4',
        'banana_action1_trial5',
        'banana_action1_trial6',
        'banana_action1_trial7',
        'banana_action1_trial8',
        'banana_action1_trial9',
        'banana_action7_trialposthoc_32_online',
        'broccoli_action2_trial9',
        'broccoli_action3_trial0',
        'broccoli_action3_trial1',
        'broccoli_action3_trial2',
        'broccoli_action3_trial3',
        'broccoli_action3_trial4',
        'broccoli_action3_trial5',
        'broccoli_action3_trial6',
        'broccoli_action6_trialposthoc_117_online',
        'broccoli_action7_trialposthoc_33_online',
        'carrot_action0_trialdebug',
        'carrot_action1_trialdebug',
        'carrot_action2_trialdebug',
        'carrot_action3_trialdebug',
        'carrot_action4_trial7',
        'carrot_action4_trial8',
        'carrot_action4_trial9',
        'carrot_action4_trialdebug',
        'carrot_action4_trialposthoc_126_online',
        'carrot_action5_trialdebug',
        'carrot_action6_trialdebug',
        'carrot_action7_trialdebug',
        'carrot_action8_trialdebug',
        'donut_action10_trial9',
        'donut_action13_trial0',
        'donut_action13_trial1',
        'donut_action13_trial2',
        'donut_action13_trialposthoc_41_online',
        'donut_action4_trialposthoc_125_online',
        'donut_action7_trial0',
        'donut_action7_trial1',
        'donut_action7_trial2',
        'donut_action7_trial3',
        'donut_action7_trial4',
        'donut_action7_trial5',
        'fries_action10_trial1',
        'fries_action10_trial2',
        'fries_action10_trial3',
        'fries_action10_trial4',
        'fries_action10_trial5',
        'fries_action10_trial6',
        'fries_action10_trial7',
        'fries_action10_trial8',
        'fries_action10_trial9',
        'fries_action4_trialposthoc_124_online',
        'fries_action5_trialposthoc_40_online',
        'fries_action7_trial4',
        'fries_action7_trial5',
        'fries_action7_trialdebug',
        'grape_action11_trialvideo/result.yaml',
        'grape_action3_trialposthoc_30_online',
        'grape_action4_trialposthoc_128_online',
        'grape_action4_trialposthoc_2_online/result.yaml',
        'jello_action3_trialposthoc_34_online',
        'jello_action4_trialposthoc_118_online',
        'kiwi_action4_trialposthoc_122_online',
        'kiwi_action7_trialposthoc_38_online',
        'none_action8_trialdebug',
        'noodles_action0_trial0',
        'noodles_action0_trial1',
        'noodles_action0_trial2',
        'noodles_action0_trial3',
        'noodles_action0_trial4',
        'noodles_action0_trial5',
        'noodles_action0_trial6',
        'noodles_action0_trial7',
        'noodles_action0_trial8',
        'noodles_action0_trial9',
        'noodles_action11_trialposthoc_36_online',
        'noodles_action1_trial0',
        'noodles_action1_trial1',
        'noodles_action1_trial2',
        'noodles_action1_trial3',
        'noodles_action1_trial4',
        'noodles_action1_trial5',
        'noodles_action1_trial6',
        'noodles_action1_trial7',
        'noodles_action1_trial8',
        'noodles_action1_trial9',
        'noodles_action2_trial0',
        'noodles_action2_trial1',
        'noodles_action2_trial2',
        'noodles_action2_trial3',
        'noodles_action2_trial4',
        'noodles_action2_trial5',
        'noodles_action2_trial6',
        'noodles_action2_trial7',
        'noodles_action2_trial8',
        'noodles_action2_trial9',
        'noodles_action3_trial0',
        'noodles_action3_trial1',
        'noodles_action3_trial2',
        'noodles_action3_trial3',
        'noodles_action3_trial4',
        'noodles_action3_trial5',
        'noodles_action3_trial6',
        'noodles_action3_trial7',
        'noodles_action4_trial2',
        'noodles_action4_trial3',
        'noodles_action4_trial4',
        'noodles_action4_trial5',
        'noodles_action4_trial6',
        'noodles_action4_trial7',
        'noodles_action4_trial8',
        'noodles_action4_trial9',
        'noodles_action5_trial0',
        'noodles_action5_trial1',
        'noodles_action5_trial2',
        'noodles_action5_trial3',
        'noodles_action5_trial4',
        'noodles_action5_trial5',
        'noodles_action5_trial6',
        'noodles_action5_trial7',
        'noodles_action5_trial8',
        'noodles_action5_trial9',
        'noodles_action5_trialposthoc_120_online',
        'noodles_action9_trial0',
        'potato_action0_trial0',
        'potato_action0_trial1',
        'potato_action0_trial2',
        'potato_action0_trial3',
        'potato_action0_trial4',
        'potato_action0_trial5',
        'potato_action0_trial6',
        'potato_action0_trial7',
        'potato_action0_trial8',
        'potato_action0_trial9',
        'potato_action12_trialposthoc_121_online',
        'potato_action1_trial0',
        'potato_action1_trial1',
        'potato_action1_trial2',
        'potato_action1_trial3',
        'potato_action1_trial4',
        'potato_action1_trial5',
        'potato_action1_trial6',
        'potato_action1_trial7',
        'potato_action1_trial8',
        'potato_action1_trial9',
        'potato_action2_trial0',
        'potato_action2_trial1',
        'potato_action2_trial2',
        'potato_action2_trial3',
        'potato_action2_trial4',
        'potato_action2_trial5',
        'potato_action2_trial6',
        'potato_action6_trialposthoc_37_online',
        'rice_action10_trial0',
        'rice_action10_trial1',
        'rice_action10_trial2',
        'rice_action11_trial7',
        'rice_action11_trial8',
        'rice_action11_trial9',
        'rice_action12_trial0',
        'rice_action12_trial1',
        'rice_action12_trial2',
        'rice_action12_trial3',
        'rice_action12_trial4',
        'rice_action12_trial5',
        'rice_action12_trial6',
        'rice_action12_trial7',
        'rice_action12_trial8',
        'rice_action12_trial9',
        'rice_action13_trial0',
        'rice_action13_trial1',
        'rice_action3_trial8',
        'rice_action3_trial9',
        'rice_action4_trial0',
        'rice_action4_trial1',
        'rice_action4_trial2',
        'rice_action4_trial3',
        'rice_action4_trial4',
        'rice_action4_trial5',
        'rice_action4_trial6',
        'rice_action4_trial7',
        'rice_action4_trial8',
        'rice_action6_trialposthoc_119_online',
        'rice_action7_trialposthoc_35_online',
        'rice_action8_trial3',
        'rice_action8_trial4',
        'rice_action8_trial5',
        'rice_action8_trial6',
        'rice_action8_trial7',
        'rice_action8_trial8',
        'rice_action8_trial9',
        'rice_action9_trial0',
        'rice_action9_trial1',
        'rice_action9_trial3',
        'rice_action9_trial4',
        'rice_action9_trial5',
        'rice_action9_trial6',
        'rice_action9_trial7',
        'rice_action9_trial8',
        'rice_action9_trial9',
        'sandwich_action4_trialposthoc_123_online',
        'sandwich_action7_trialposthoc_39_online',
        'spinach_action10_trial3',
        'spinach_action12_trialposthoc_1_online/result.yaml',
        'spinach_action13_trial5/result.yaml',
        'spinach_action4_trialposthoc_127_online',
        'strawberry_action4_trial2',
        'strawberry_action4_trial3',
        'strawberry_action4_trial4',
        'strawberry_action4_trial5',
        'strawberry_action4_trial6',
        'strawberry_action4_trial7',
        'strawberry_action4_trial8',
        'strawberry_action4_trial9',
        'strawberry_action5_trial0',
        'strawberry_action5_trial1',
        'strawberry_action5_trial2',
        'strawberry_action5_trial3',
        'strawberry_action5_trial4',
        'strawberry_action5_trial5',
        'strawberry_action5_trial6',
        'strawberry_action5_trial7',
        'strawberry_action5_trial8',
        'strawberry_action5_trial9',
        'strawberry_action6_trial0',
        'strawberry_action6_trial1',
        'strawberry_action6_trial2',
        'strawberry_action6_trial3',
        'strawberry_action6_trial4',
        'strawberry_action6_trial5',
        'strawberry_action6_trial6',
        'strawberry_action6_trial7',
        'strawberry_action6_trial8',
        'strawberry_action6_trial9',
        'strawberry_action7_trialposthoc_31_online'])

    # Get all the rosbags in rss_experiments_folder_path along with their
    # success status
    rosbags_and_success: List[Tuple[str, bool]] = get_all_rosbags_and_success(
        rss_experiments_folder_path, rosbags_to_ignore=rosbags_to_ignore
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
