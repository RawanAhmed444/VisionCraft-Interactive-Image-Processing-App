import numpy as np
import cv2
from src.utils import convert_to_grayscale, convolve, magnitude
from src.functions.edge_functions import smooth_image, compute_sobel_gradients, non_maximum_suppression, apply_double_thresholding, apply_hysteresis

class EdgeProcessor:
    """
    Processes edges in an image using modular functions for different methods:
    Sobel, Prewitt, Roberts, and Canny.
    """
    def __init__(self, image):
        """
        Initializes the processor with an image.
        
        :param image: Input image in BGR or grayscale.
        """
        # Convert to grayscale if needed
        self.image = convert_to_grayscale(image)
        self.filtered_image = None  # For smoothed image
        self.edge_map = None

    def apply_filter(self, filter_type="gaussian", kernel_size=5, sigma=1.4):
        """
        Applies a smoothing filter to the image.
        
        :param filter_type: Currently supports only 'gaussian'.
        :param kernel_size: Size of the kernel.
        :param sigma: Standard deviation for the Gaussian filter.
        """
        if filter_type == "gaussian":
            self.filtered_image = smooth_image(self.image, kernel_size, sigma)
        else:
            raise ValueError("Unsupported filter type. Use 'gaussian'.")
        return self.filtered_image

    def sobel_edge_detection(self):
        """
        Applies Sobel edge detection.
        
        :return: Edge map (normalized to 0-255).
        """
        if self.filtered_image is None:
            # Optionally, apply default Gaussian smoothing
            self.apply_filter(filter_type="gaussian", kernel_size=3, sigma=1.0)
        
        # Compute Sobel gradients using our modular function
        Gx, Gy = compute_sobel_gradients(self.filtered_image)
        
        # Compute gradient magnitude
        G = magnitude(Gx, Gy)
        
        # Normalize for visualization
        self.edge_map = np.uint8(255 * G / np.max(G))
        return self.edge_map

    def canny_edge_detection(self, low_threshold=None, high_threshold=None):
        """
        Full Canny edge detection pipeline:
          1. Gaussian smoothing
          2. Compute gradients (using Sobel kernels)
          3. Compute gradient magnitude and angle
          4. Non-maximum suppression
          5. Double thresholding and hysteresis
          
        :param low_threshold: Lower threshold for double thresholding.
        :param high_threshold: Higher threshold for double thresholding.
        :return: Final edge map.
        """
        # Step 1: Gaussian smoothing (if not already filtered)
        smooth = self.filtered_image if self.filtered_image is not None else smooth_image(self.image, 5, 1.4)
        
        # Step 2: Compute gradients
        Gx, Gy = compute_sobel_gradients(smooth)
        G = magnitude(Gx, Gy)
        
        # Step 3: Compute gradient direction
        theta = np.arctan2(Gy, Gx) * 180 / np.pi
        theta[theta < 0] += 180  # Normalize angles to [0, 180]
        
        # Step 4: Non-Maximum Suppression
        nms = non_maximum_suppression(G, theta)
        
        # Step 5: Double Thresholding (set thresholds relative to max if not provided)
        max_val = np.max(nms)
        low_threshold = low_threshold or (max_val * 0.1)
        high_threshold = high_threshold or (max_val * 0.5)
        strong_edges, weak_edges = apply_double_thresholding(nms, low_threshold, high_threshold)
        
        # Step 6: Hysteresis: link weak edges to strong edges
        self.edge_map = apply_hysteresis(strong_edges, weak_edges)
        
        return self.edge_map

    def prewitt_edge_detection(self):
        """
        Applies Prewitt edge detection.
        
        :return: Edge map.
        """
        # Define Prewitt kernels
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]])
        
        Gx = convolve(self.image, kernel_x)
        Gy = convolve(self.image, kernel_y)
        self.edge_map = magnitude(Gx, Gy)
        return self.edge_map

    def roberts_edge_detection(self):
        """
        Applies Roberts edge detection.
        
        :return: Edge map.
        """
        # Define Roberts kernels
        kernel_x = np.array([[1, 0],
                             [0, -1]])
        kernel_y = np.array([[0, 1],
                             [-1, 0]])
        
        Gx = convolve(self.image, kernel_x)
        Gy = convolve(self.image, kernel_y)
        self.edge_map = magnitude(Gx, Gy)
        return self.edge_map

    def get_edge_map(self):
        """Returns the computed edge map."""
        if self.edge_map is None:
            raise ValueError("Edge map not computed. Run an edge detection method first.")
        return self.edge_map
