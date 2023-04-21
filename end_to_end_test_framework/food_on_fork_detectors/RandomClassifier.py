from .FoodOnForkInterface import FoodOnForkInterface
import numpy as np


class RandomClassifier(FoodOnForkInterface):
    def __init__(self, seed: int = 0):
        """
        Initializes the random classifier.

        Parameters
        ----------
        seed : int the seed to use for the random number generator
        """
        # Initialize the random number generator
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        """
        Called right before a bite acquisition attempt. This function should
        reset any state that was maintained from the previous bite acquisition
        attempt.
        """
        # Do nothing
        return

    def next_image(self, img):
        """
        Called when the camera captures a new frame in the bite acquisition
        attempt.
        """
        # Do nothing
        return

    def predict(self):
        """
        Called at the end of a bite acquisition attempt. Should return a
        boolean, whether there is food on the fork or not.
        """
        # Return a random prediction
        return self.rng.random() >= 0.5
