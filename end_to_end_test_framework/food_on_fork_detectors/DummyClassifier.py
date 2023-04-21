from .FoodOnForkInterface import FoodOnForkInterface


class DummyClassifier(FoodOnForkInterface):
    def __init__(self, pred: bool):
        """
        Initializes the dummy classifier.

        Parameters
        ----------
        pred : bool the prediction to always return
        """
        self.pred = pred

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
        # Always return self.pred
        return self.pred
