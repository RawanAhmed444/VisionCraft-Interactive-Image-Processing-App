import cv2
import numpy as np
from functions.edge_functions import (
    sobel_edge_detection,
    canny_edge_detection,
    prewitt_edge_detection,
    roberts_edge_detection,
    smooth_image
)

from functions.hough_transform_functions import(
    detect_lines,
    hough_line_detection,
    detect_circles,
    hough_circle_detection,
    hough_ellipse_detection
    
)
from utils import convert_to_grayscale,get_dimensions, convolve

class EdgeDetector:
    """
    A class to perform edge detection using various algorithms.
    It uses modular functions from edge_functions for actual processing.
    """
    def __init__(self):
        """
        Initializes the EdgeDetector with an input image.
        
        :param image: Input image in BGR or grayscale format.
        """
        # Convert to grayscale if necessary
        self.image = None
        # Dictionary to store edge maps computed by various methods
        self.edge_maps = {}

    def set_image(self, image):
        """
        Sets the input image for edge detection.
        
        :param image: Input image in BGR or grayscale format.
        """
        # Convert to grayscale if necessary
        self.image = convert_to_grayscale(image)
        self.edge_maps = {}

            
    def detect_edges(self, detector_method='sobel', **kwargs):
        """
        Detects edges using the specified method.
        
        :param detector_method: String indicating the method ('sobel', 'canny', 'prewitt', 'roberts').
        :param kwargs: Additional arguments for edge detection.
        :return: Edge map produced by the specified method.
        """
        if detector_method == 'sobel':
            # Filter kwargs to only include valid arguments for detect_sobel
            valid_kwargs = {k: v for k, v in kwargs.items() if k in ['kernel_size', 'sigma']}
            return self.detect_sobel(**valid_kwargs)
        elif detector_method == 'canny':
            # Filter kwargs to only include valid arguments for detect_canny
            valid_kwargs = {k: v for k, v in kwargs.items() if k in ['low_threshold', 'high_threshold', 'max_edge_val', 'min_edge_val']}
            return self.detect_canny(**valid_kwargs)
        elif detector_method == 'prewitt':
            # Filter kwargs to only include valid arguments for detect_prewitt
            valid_kwargs = {k: v for k, v in kwargs.items() if k in ['threshold', 'value']}
            return self.detect_prewitt(**valid_kwargs)
        elif detector_method == 'roberts':
            # No kwargs for detect_roberts
            return self.detect_roberts()
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

     
        def magnitude(Gx, Gy):
            """Computes the gradient magnitude."""
            return np.hypot(Gx, Gy)

        def non_maximum_suppression(G, theta):
            height, width = G.shape
            nms = np.zeros_like(G, dtype=np.uint8)
            theta = theta % 180

            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    angle = theta[y, x]
                    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                        p1, p2 = G[y, x-1], G[y, x+1]
                    elif 22.5 <= angle < 67.5:
                        p1, p2 = G[y-1, x+1], G[y+1, x-1]
                    elif 67.5 <= angle < 112.5:
                        p1, p2 = G[y-1, x], G[y+1, x]
                    elif 112.5 <= angle < 157.5:
                        p1, p2 = G[y-1, x-1], G[y+1, x+1]

                    if G[y, x] >= p1 and G[y, x] >= p2:
                        nms[y, x] = G[y, x]

            return nms

        def apply_double_thresholding(image, low_threshold, high_threshold, max_edge_val=255, min_edge_val=0):
            strong_edges = np.zeros_like(image)
            weak_edges = np.zeros_like(image)

            strong_edges[image >= high_threshold] = max_edge_val
            weak_edges[(image >= low_threshold) & (image < high_threshold)] = min_edge_val
            return strong_edges, weak_edges

        def apply_hysteresis(strong_edges, weak_edges, max_edge_val=255):
            height, width = strong_edges.shape
            final_edges = np.copy(strong_edges)
            stack = list(zip(*np.where(strong_edges == max_edge_val)))

            while stack:
                y, x = stack.pop()
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and weak_edges[ny, nx] != 0:
                            final_edges[ny, nx] = max_edge_val
                            weak_edges[ny, nx] = 0
                            stack.append((ny, nx))

            return final_edges

        def compute_sobel_gradients(image):
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            Gx = convolve(image, kernel_x)
            Gy = convolve(image, kernel_y)
            return Gx, Gy

        def canny_edge_detection(image, low_threshold=None, high_threshold=None, max_edge_val=255, min_edge_val=0):
            smooth = smooth_image(image)
            Gx, Gy = compute_sobel_gradients(smooth)
            G = magnitude(Gx, Gy)
            G_normalized = np.uint8(G / G.max() * 255)

            theta = np.rad2deg(np.arctan2(Gy, Gx)) % 180
            nms = non_maximum_suppression(G_normalized, theta)

            max_val = np.max(nms)
            low_threshold = low_threshold if low_threshold is not None else max_val * 0.1
            high_threshold = high_threshold if high_threshold is not None else max_val * 0.5

            strong_edges, weak_edges = apply_double_thresholding(nms, low_threshold, high_threshold, max_edge_val, min_edge_val)
            final_edges = apply_hysteresis(strong_edges, weak_edges)

            return final_edges

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

    def detect_shape(self, shape_type = 'line', **kwargs):
        if shape_type == 'line':
            valid_kwargs = {k: v for k, v in kwargs.items() if k in ['num_rho', 'num_theta', 'blur_ksize', 'low_threshold', 'high_threshold', 'hough_threshold_ratio']}
            return hough_line_detection(self.image, **valid_kwargs)
        elif shape_type == 'circle':
            valid_kwargs = {k: v for k, v in kwargs.items() if k in ['r_min', 'r_max', 'delta_r', 'num_thetas','blur_ksize', 'min_edge_threshold', 'max_edge_threshold', 'bin_threshold']}
            return hough_circle_detection(self.image, **valid_kwargs)
        elif shape_type == 'ellipse':
            valid_kwargs =  {k: v for k, v in kwargs.items() if k in ['min_d', 'max_d', 'step_size', 'blur_ksize', 'low_threshold', 'high_threshold', 'hough_threshold_ratio']}
            return hough_ellipse_detection(self.image, **valid_kwargs)

    
    def detect_lines(self,  **kwargs):
        
        self.edge_maps['canny'] = self.edge_maps.get('canny') or self.detect_edges(detector_method='canny', **kwargs)
        edged_image = self.edge_maps['canny']
        
        
        h, w = get_dimensions(self.image)
        diagonal = int(np.ceil(np.sqrt(h**2 + w**2)))
        
        dtheta = 180 / kwargs['num_theta'] 
        drho = 2 * diagonal // kwargs['num_rho']
        
        # Create theta and rho ranges
        thetas = np.arange(0, 180, step=dtheta)
        rhos = np.arange(-diagonal, diagonal, step=drho)
        accumulator = np.zeros((len(rhos), len(thetas)))
        
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))        

        
        # Hough Transform
        for y in range(h):
            for x in range(w):
                if edged_image[y, x] != 0:
                    center = [y - h//2, x - w//2]
                    for theta_idx in range(len(thetas)):
                        rho = (center[1] * cos_thetas[theta_idx]) + (center[0] * sin_thetas[theta_idx])
                        rho_idx = np.argmin(abs(rhos - rho))
                        accumulator[rho_idx][theta_idx] += 1
        
        max_val = np.max(accumulator)
        threshold = max_val * kwargs['hough_threshold_ratio']
        output_image = self.image.copy()
        for y in range(accumulator.shape[0]):
            for x in range(accumulator.shape[1]):
                if accumulator[y][x] > threshold:
                    rho = rhos[y]
                    theta = thetas[x]
                    a = np.cos(np.deg2rad(theta))
                    b = np.sin(np.deg2rad(theta))
                    x0 = (a * rho) + (w//2)
                    y0 = (b * rho) + (h//2)
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    output_image = cv2.line(output_image, (x1,y1), (x2,y2), (0,255,0), 1)
                    
        return output_image
    
    def detect_circles(self, **kwargs):
        
        self.edge_maps['canny'] = self.edge_maps.get('canny') or self.detect_edges(detector_method='canny', **kwargs)
        edged_image = self.edge_maps['canny']
        
         
        h, w = get_dimensions(self.image)
        r_max = kwargs['r_max'] or (min(h, w) // 2)
        
        # R and Theta ranges
        dtheta = int(360 / kwargs['num_thetas'])
        thetas = np.arange(0, 360, step=dtheta)
        rs = np.arange(kwargs['r_min'], r_max, step=kwargs['delta_r'])

        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))
        
        
        
                   
    
            
    
            
            
            
