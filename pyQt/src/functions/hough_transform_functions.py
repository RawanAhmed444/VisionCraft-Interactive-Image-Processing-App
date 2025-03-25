import cv2
import numpy as np
import math
from collections import defaultdict
from utils import get_dimensions
from functions.edge_functions import canny_edge_detection

def detect_lines(image = None, num_rho=180, num_theta=180, blur_ksize=5, low_threshold=50, high_threshold=150, hough_threshold_ratio=0.6):
    '''
    Applies the Hough Transform to detect lines in the image.
    args:
        image: Input image (NumPy array).
        num_rho: Number of rho values.
        num_theta: Number of theta values.
        blur_ksize: Kernel size for Gaussian blur.
        low_threshold: Low threshold for Canny edge detection.
        high_threshold: High threshold for Canny edge detection.
        hough_threshold_ratio: Threshold ratio for Hough Transform.
    returns:
        Output image with detected lines.
    '''
    
    # Apply Canny Edge Detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured_image = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)
    edged_image = canny_edge_detection(blured_image)

    # Image dimensions
    height, width = edged_image.shape[:2]
    half_height, half_width = height / 2, width / 2

    # Calculate diagonal length and steps (resolution)
    diagonal = int(math.sqrt(height**2 + width**2))
    dtheta = 180 / num_theta
    drho = (2 * diagonal) / num_rho

    # Create theta and rho ranges
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-diagonal, diagonal, step=drho)
    
    # Initialize accumulator
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    # Precompute trig values
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    # Hough Transform
    for y in range(height):
        for x in range(width):
            if edged_image[y, x] != 0:
                center = [y - half_height, x - half_width]
                for theta_idx in range(len(thetas)):
                    rho = (center[1] * cos_thetas[theta_idx]) + (center[0] * sin_thetas[theta_idx])
                    rho_idx = np.argmin(abs(rhos - rho))
                    accumulator[rho_idx][theta_idx] += 1

    # Find max value in accumulator
    max_val = np.max(accumulator)
    threshold = max_val * hough_threshold_ratio

    # Detect lines from accumulator
    output_image = image.copy()
    for y in range(accumulator.shape[0]):
        for x in range(accumulator.shape[1]):
            if accumulator[y][x] > threshold:
                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho) + half_width
                y0 = (b * rho) + half_height
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                output_image = cv2.line(output_image, (x1,y1), (x2,y2), (0,255,0), 1)
    
    return output_image


def get_dimensions(image):
    """Returns height and width of the image."""
    return image.shape[:2]

def detect_edges(image, method="canny", **kwargs):
    """Detects edges in an image using the specified method."""
    if method == "canny":
        return canny_edge_detection(image)
    raise ValueError("Unsupported edge detection method.")

def initialize_hough_space(image, num_theta=180, num_rho=None):
    """Initializes the Hough space parameter grid."""
    h, w = get_dimensions(image)
    max_rho = int(np.ceil(np.sqrt(h**2 + w**2)))
    num_rho = num_rho if num_rho else 2 * max_rho
    
    thetas = np.deg2rad(np.linspace(0, 180, num_theta))
    rhos = np.linspace(-max_rho, max_rho, num_rho)

    accumulator = np.zeros((num_rho, num_theta), dtype=np.int32)
    return thetas, rhos, accumulator

def compute_hough_votes(edges, thetas, rhos, accumulator):
    """Computes the votes for the Hough accumulator."""
    h, w = edges.shape
    edge_pixels = np.argwhere(edges > 0)

    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    for y, x in edge_pixels:
        rhos_calc = x * cos_thetas + y * sin_thetas
        rho_indices = np.digitize(rhos_calc, rhos) - 1
        valid_indices = (rho_indices >= 0) & (rho_indices < len(rhos))
        
        accumulator[rho_indices[valid_indices], np.where(valid_indices)[0]] += 1

    return accumulator

def extract_hough_peaks(accumulator, rhos, thetas, threshold_ratio=0.5, min_distance=10):
    """Extracts peaks from the Hough accumulator using thresholding and non-maximum suppression."""
    max_votes = np.max(accumulator)
    threshold = threshold_ratio * max_votes
    peak_indices = np.argwhere(accumulator > threshold)

    # Apply Non-Maximum Suppression (NMS)
    peaks = []
    for rho_idx, theta_idx in peak_indices:
        rho, theta = rhos[rho_idx], thetas[theta_idx]
        
        if all(abs(rho - prho) > min_distance for prho, _ in peaks):
            peaks.append((rho, theta))
    
    return peaks

def filter_lines_by_angle(lines, min_angle=10, max_angle=170):
    """Filters lines to remove near-horizontal and near-vertical ones."""
    filtered_lines = []
    for rho, theta in lines:
        angle = np.rad2deg(theta)
        if min_angle < abs(angle) < max_angle:
            filtered_lines.append((rho, theta))
    return filtered_lines

def draw_detected_lines(image, edge, detected_lines):
    """Draws detected lines on the original image."""
    output_image = image.copy()
    h, w = get_dimensions(edge)

    for rho, theta in detected_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = (a * rho)
        y0 = (b * rho) 
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        output_image = cv2.line(output_image, (x1,y1), (x2,y2), (0,255,0), 1)
        
    return output_image

def hough_line_detection(image, **kwargs):
    """Performs full Hough line detection."""
    edge_param =  {k: v for k, v in kwargs.items() if k in ['low_threshold', 'high_threshold', 'max_edge_val', 'min_edge_val']}
    init_param = {k: v for k, v in kwargs.items() if k in ['num_theta', 'num_rho']}

    edges = detect_edges(image, **edge_param)
    thetas, rhos, accumulator = initialize_hough_space(edges, **init_param)
    accumulator = compute_hough_votes(edges, thetas, rhos, accumulator)
    detected_lines = extract_hough_peaks(accumulator, rhos, thetas, kwargs.get('hough_threshold_ratio', 0.5))
    filtered_lines = filter_lines_by_angle(detected_lines)

    return draw_detected_lines(image, edges,filtered_lines)

def initialize_hough_circle_space(image, r_min, r_max):
    """Initializes the 3D Hough space for circle detection."""
    h, w = image.shape
    print(r_min, r_max)
    accumulator = np.zeros((h, w, abs(r_max - r_min)), dtype=np.int32)
    return accumulator

def compute_hough_circle_votes(edges, accumulator, radius_range):
    """Computes the Hough votes for circle detection."""
    h, w = edges.shape
    min_radius, max_radius = radius_range
    edge_pixels = np.argwhere(edges > 0)

    theta = np.deg2rad(np.arange(0, 360, 5))  # Precompute angles
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for y, x in edge_pixels:
        for r in range(min_radius, max_radius):
            a = (x - r * cos_t).astype(int)  # Compute all a values at once
            b = (y - r * sin_t).astype(int)  # Compute all b values at once
            
            valid_idx = (0 <= a) & (a < w) & (0 <= b) & (b < h)  # Filter valid indices
            np.add.at(accumulator, (b[valid_idx], a[valid_idx], r - min_radius), 1)
    return accumulator

def non_maximum_suppression(circles, accumulator, min_dist=40, r_min = 10):
    """Applies Non-Maximum Suppression to remove overlapping circles based on Euclidean distance."""
    filtered_circles = []
    
    # Sort circles by vote strength (higher votes first)
    sorted_circles = sorted(circles, key=lambda c: -accumulator[c[1], c[0], c[2] - r_min])
    
    for x, y, r in sorted_circles:
        keep = True
        for x2, y2, r2 in filtered_circles:
            dist = np.linalg.norm(np.array((x, y)) - np.array((x2, y2)))  # Euclidean distance
            if dist < min_dist + abs(r - r2):  # Consider radius difference
                keep = False
                break  # Skip adding this circle if it's too close
        if keep:
            filtered_circles.append((x, y, r))
    
    return filtered_circles

def extract_circle_peaks(accumulator, radius_range, threshold_ratio=0.5, min_dist = 20):
    """Extracts circle peaks from the 3D accumulator."""
    min_radius, _ = radius_range
    max_votes = np.max(accumulator)
    threshold = threshold_ratio * max_votes
    peak_indices = np.argwhere(accumulator > threshold)
    circles = [(a, b, r + min_radius) for b, a, r in peak_indices]

    # Apply Non-Maximum Suppression
    return non_maximum_suppression(circles, accumulator, min_dist=min_dist, r_min= min_radius)
def draw_detected_circles(image, detected_circles):
    """Draws detected circles on the original image."""
    output_image = image.copy()
    for a, b, r in detected_circles:
        cv2.circle(output_image, (a, b), r, (0, 255, 0), 2)
    return output_image

def hough_circle_detection(image, r_min=10, r_max =100, **kwargs):
    """Performs full Hough circle detection."""
    radius_range=(r_min, r_max)
    edges = detect_edges(image, **kwargs)
    accumulator = initialize_hough_circle_space(edges, r_min, r_max)
    print('accumlator done!!')
    accumulator = compute_hough_circle_votes(edges, accumulator, radius_range)
    detected_circles = extract_circle_peaks(accumulator, radius_range, kwargs.get('bin_threshold', 0.5))
    drawn = draw_detected_circles(image, detected_circles)
    return detected_circles


# 🔹 Step 1: Edge Detection (Same as Circle Detection)
def detect_edges(image, method="canny", **kwargs):
    """Detects edges in an image using the specified method."""
    if method == "canny":
        return canny_edge_detection(image)
    raise ValueError("Unsupported edge detection method.")

# 🔹 Step 2: Initialize Hough Accumulator for Ellipse Detection
def initialize_hough_ellipse_space(image, a_min, a_max, b_min, b_max, theta_step=10):
    """Initializes the 5D Hough space for ellipse detection."""
    h, w = image.shape
    a_range = range(a_min, a_max, 5)  # Step 10 to reduce memory
    b_range = range(b_min, b_max, 5)
    theta_range = range(0, 180, theta_step)  # Reduce angular resolution

    accumulator = np.zeros((h, w, len(a_range), len(b_range), len(theta_range)), dtype=np.int32)
    return accumulator, list(a_range), list(b_range), list(theta_range)

# 🔹 Step 3: Compute Votes for Ellipses
def compute_hough_ellipse_votes(edges, accumulator, a_range, b_range, theta_values):
    """Computes votes for the Hough Transform ellipse detection."""
    h, w = edges.shape
    edge_pixels = np.argwhere(edges > 0)

    cos_t, sin_t = np.cos(np.deg2rad(theta_values)), np.sin(np.deg2rad(theta_values))

    for y, x in edge_pixels:
        for a_idx, a in enumerate(a_range):
            for b_idx, b in enumerate(b_range):
                for t_idx, (cos_theta, sin_theta) in enumerate(zip(cos_t, sin_t)):
                    a0_new = int(x - a * cos_theta)  # Compute ellipse center X
                    b0_new = int(y - b * sin_theta)  # Compute ellipse center Y

                    # Ensure indices are within bounds
                    if 0 <= a0_new < w and 0 <= b0_new < h:
                        np.add.at(accumulator, (b0_new, a0_new, a_idx, b_idx, t_idx), 1)

    return accumulator

# 🔹 Step 4: Extract Peaks using Non-Maximum Suppression
def extract_ellipse_peaks(accumulator, a_range, b_range, theta_values, threshold_ratio=0.5, min_dist=20):
    """Extracts ellipse peaks from the 5D accumulator."""
    max_votes = np.max(accumulator)
    threshold = threshold_ratio * max_votes
    peak_indices = np.argwhere(accumulator > threshold)

    detected_ellipses = []
    for b0, a0, a_idx, b_idx, t_idx in peak_indices:
        a, b, theta = a_range[a_idx], b_range[b_idx], theta_values[t_idx]

        # Non-max suppression to avoid overlapping ellipses
        if all(np.linalg.norm(np.array([a0, b0]) - np.array([x, y])) > min_dist for x, y, _, _, _ in detected_ellipses):
            detected_ellipses.append((a0, b0, a, b, theta))

    return detected_ellipses

# 🔹 Step 5: Draw Detected Ellipses
def draw_detected_ellipses(image, detected_ellipses):
    """Draws detected ellipses on the original image."""
    output_image = image.copy()
    for a0, b0, a, b, theta in detected_ellipses:
        cv2.ellipse(output_image, (a0, b0), (a, b), theta, 0, 360, (0, 255, 0), 2)
    return output_image

# 🔹 Step 6: Full Hough Ellipse Detection Pipeline
def hough_ellipse_detection(image, a_min=20, a_max=100, b_min=10, b_max=50, **kwargs):
    """Performs full Hough ellipse detection."""
    edges = detect_edges(image, **kwargs)
    accumulator, a_range, b_range, theta_values = initialize_hough_ellipse_space(edges, a_min, a_max, b_min, b_max)
    accumulator = compute_hough_ellipse_votes(edges, accumulator, a_range, b_range, theta_values)
    detected_ellipses = extract_ellipse_peaks(accumulator, a_range, b_range, theta_values, kwargs.get('hough_threshold_ratio', 0.5))
    return draw_detected_ellipses(image, detected_ellipses)


def detect_circles(image, min_edge_threshold=50, max_edge_threshold=150, r_min=20, r_max=None, delta_r=1, num_thetas=50, bin_threshold=0.4):
    '''
    Detects circles in the image using Hough Transform.
    args:
        image: Input image (NumPy array).
        min_edge_threshold: Minimum edge threshold for Canny edge detection.
        max_edge_threshold: Maximum edge threshold for Canny edge detection.
        r_min: Minimum radius of the circles.
        r_max: Maximum radius of the circles.
        delta_r: Delta radius for the circles.
        num_thetas: Number of theta values.
        bin_threshold: Threshold for the accumulator bins.
    returns:
        Output image with detected circles.
    '''
    
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blured_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edged_image = canny_edge_detection(blured_image)

        height, width = image.shape[:2]
        if r_max is None:
            r_max = min(height, width) // 2

        # R and Theta ranges
        dtheta = int(360 / num_thetas)
        thetas = np.arange(0, 360, step=dtheta)
        rs = np.arange(r_min, r_max, step=delta_r)

        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        # Evaluate and keep ready the candidate circles dx and dy for different delta radius
        circle_candidates = []
        for r in rs:
            for t in range(num_thetas):
                circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))

        circle_candidates = np.array(circle_candidates)

        accumulator = defaultdict(int)

        for y in range(height):
            for x in range(width):
                # Check for white pixels
                if edged_image[y][x] != 0: 
                    # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                    for r, rcos_t, rsin_t in circle_candidates:
                        x_center = x - rcos_t
                        y_center = y - rsin_t
                        accumulator[(x_center, y_center, r)] += 1 

        output_img = image.copy()
        # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
        out_circles = []
        # Sort the accumulator descendingly based on the votes for the candidate circles 
        sorted_accumulator = sorted(accumulator.items(), key=lambda i: -i[1])

        for candidate_circle, votes in sorted_accumulator:
            x, y, r = candidate_circle
            current_vote_percentage = votes / num_thetas
            if current_vote_percentage >= bin_threshold: 
                # Shortlist the circle for final result
                out_circles.append((x, y, r, current_vote_percentage))

        post_process = True
        if post_process:
            pixel_threshold = 5
            postprocess_circles = []
            for x, y, r, v in out_circles:
                if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                    postprocess_circles.append((x, y, r, v))
            out_circles = postprocess_circles

        for x, y, r, v in out_circles:
            output_img = cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)
        
        return output_img
    except Exception as e:
        print(f"Error in detect_circles: {e}")
        return image