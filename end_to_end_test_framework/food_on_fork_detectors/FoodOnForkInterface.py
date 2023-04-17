class FoodOnForkInterface:
    def initialize(self):
        """
        Called right before a bite acquisition attempt. This function should
        reset any state that was maintained from the previous bite acquisition
        attempt.
        """
        raise NotImplementedError()

    def next_image(self, img):
        """
        Called when the camera captures a new frame in the bite acquisition
        attempt.
        """
        raise NotImplementedError()

    def predict(self):
        """
        Called at the end of a bite acquisition attempt. Should return a
        boolean, whether there is food on the fork or not.
        """
        raise NotImplementedError()
