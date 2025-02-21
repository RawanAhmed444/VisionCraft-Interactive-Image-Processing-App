import numpy as np
from functions.thresholding_functions import globalthresholding, localthresholding

class ThresholdingProcessor:
    """Applies global or local thresholding on an image."""
    
    def __init__(self, threshold_type="global", T=128, kernel=4, k=2):
        """
        Initializes the thresholding processor.

        :param threshold_type: "global" for global thresholding, "local" for local.
        :param T: Global threshold value (default = 128).
        :param kernel: Kernel size for local thresholding (default = 4).
        :param k: Weighting factor for local thresholding (default = 2).
        """
        self.image = None
        self.threshold_type = threshold_type
        self.T = T
        self.kernel = kernel
        self.k = k
        self.binary_image = None

    def apply_thresholding(self):
        """
        Applies either global or local thresholding based on the chosen type.
        
        :return: Binary image after thresholding.
        """
        if self.image is None:
            raise ValueError("No image set. Please call set_image() first.")
        
        if self.threshold_type == "global":
            self.binary_image = globalthresholding(self.image, self.T)
        elif self.threshold_type == "local":
            self.binary_image = localthresholding(self.image, self.kernel, self.k)
        else:
            raise ValueError("Invalid thresholding type. Choose 'global' or 'local'.")
        
        return self.binary_image

    def set_image(self, image):
        """
        Sets the image for thresholding.

        :param image: Input image as a NumPy array.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image input. Expected a NumPy array.")
        
        self.image = image

    def get_binary_image(self):
        """
        Returns the thresholded binary image.
        
        :return: Binary image as a NumPy array.
        """
        if self.binary_image is None:
            raise ValueError("No thresholded image available. Apply thresholding first.")
        return self.binary_image
