import cv2
from functions.edge_functions import (
    sobel_edge_detection,
    canny_edge_detection,
    prewitt_edge_detection,
    roberts_edge_detection
)
from utils import convert_to_grayscale

class EdgeDetector:
    """
    A class to perform edge detection using various algorithms.
    It uses modular functions from edge_functions for actual processing.
    """
    def __init__(self, image):
        """
        Initializes the EdgeDetector with an input image.
        
        :param image: Input image in BGR or grayscale format.
        """
        # Convert to grayscale if necessary
        self.image = convert_to_grayscale(image)
        # Dictionary to store edge maps computed by various methods
        self.edge_maps = {}

    def detect_edges(self, detector_method = 'sobel', **kwargs):
        """
        Detects edges using the specified method.
        
        :param detector_method: String indicating the method ('sobel', 'canny', 'prewitt', 'roberts').
        :param kwargs: Additional arguments for edge detection.
        :return: Edge map produced by the specified method.
        """
        if detector_method == 'sobel':
            return self.detect_sobel(**kwargs)
        elif detector_method == 'canny':
            return self.detect_canny(**kwargs)
        elif detector_method == 'prewitt':
            return self.detect_prewitt(**kwargs)
        elif detector_method == 'roberts':
            return self.detect_roberts(**kwargs)
        else:
            raise ValueError(f"Unknown edge detection method '{detector_method}'. Use 'sobel', 'canny', 'prewitt', or 'roberts'.")
    def detect_sobel(self, kernel_size=3, sigma=1.0):
        """
        Detects edges using the Sobel operator.
        
        :return: Edge map (normalized to 0-255) produced by Sobel detection.
        """
        result = sobel_edge_detection(self.image, kernel_size, sigma)
        self.edge_maps['sobel'] = result
        return result

    def detect_canny(self, low_threshold=None, high_threshold=None, max_edge_val=255, min_edge_val=0):
        """
        Detects edges using the Canny edge detector.
        
        :param low_threshold: Lower threshold for double thresholding.
        :param high_threshold: Higher threshold for double thresholding.
        :param max_edge_val: Maximum edge value (default 255).
        :param min_edge_val: Minimum edge value (default 0).
        :return: Final edge map from Canny detection.
        """
        result = canny_edge_detection(self.image, low_threshold, high_threshold, max_edge_val, min_edge_val)
        self.edge_maps['canny'] = result
        return result

    def detect_prewitt(self, threshold = 50, value = 255 ):
        """
        Detects edges using the Prewitt operator.
        
        :return: Edge map produced by Prewitt detection.
        """
        result = prewitt_edge_detection(self.image, threshold,
                                        value)
        self.edge_maps['prewitt'] = result
        return result

    def detect_roberts(self):
        """
        Detects edges using the Roberts operator.
        
        :return: Edge map produced by Roberts detection.
        """
        result = roberts_edge_detection(self.image)
        self.edge_maps['roberts'] = result
        return result

    def get_edge_map(self, method):
        """
        Retrieves the edge map for a given method.
        
        :param method: String indicating the method ('sobel', 'canny', 'prewitt', 'roberts').
        :return: Edge map if computed, otherwise None.
        """
        return self.edge_maps.get(method, None)
