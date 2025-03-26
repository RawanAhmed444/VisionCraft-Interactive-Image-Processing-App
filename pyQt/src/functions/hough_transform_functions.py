import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from io import BytesIO
from collections import defaultdict
from utils import get_dimensions
from functions.canny import canny_edge_detection, detect_edges_cv
from functions.edge_functions import canny_edge_detection_no_norm
from itertools import combinations

def lines_nms(lines, accumulator,rhos, thetas,  theta_thresh=5):
    """
    Applies NMS based on angle (theta) to remove redundant lines.

    Args:
        lines (list of tuples): List of (rho_index, theta_index) pairs.
        accumulator (numpy array): The Hough accumulator.
        thetas (numpy array): Array of theta values in degrees.
        theta_thresh (float): Angle threshold in degrees.

    Returns:
        list: Filtered list of (rho, theta) pairs.
    """
    if lines is None or len(lines) == 0:
        return []

    lines = [(rhos[idx[0]], thetas[idx[1]]) for idx in lines]
    
    lines = sorted(lines, key=lambda line: accumulator[int(np.where(rhos == line[0])[0][0]), int(np.where(thetas == line[1])[0][0])], reverse=True)

    selected_lines = []
    suppressed = set()  

    for i, (rho1, theta1) in enumerate(lines):
        if i in suppressed:
            continue
        selected_lines.append((rho1, theta1))
        for j, (rho2, theta2) in enumerate(lines[i+1:], start=i+1):
            if np.abs(float(theta1) - float(theta2)) < theta_thresh:
                suppressed.add(j) 

    return selected_lines


def visualize_lines(edges, accumulator, detected_lines, filtered_lines, rhos, thetas,output_image):
    """
    Visualizes the edge detection result, Hough Transform accumulator, and detected lines.

    Args:
        edges: The edge-detected image.
        accumulator: The Hough Transform accumulator.
        detected_lines: List of detected lines in (rho, theta) index format.
        filtered_lines: List of filtered lines in (rho, theta) value format.
        image: The original input image.
        rhos: Array of rho values.
        thetas: Array of theta values.

    Returns:
        fig: A Matplotlib figure containing the visualizations.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Plot 1: Edge Detection Result
    axs[0, 0].imshow(edges, cmap='gray')
    axs[0, 0].set_title("Edge Detection")
    axs[0, 0].axis("off")

    # Plot 2: Hough Accumulator
    axs[0, 1].imshow(accumulator, cmap='hot', extent=[thetas[0], thetas[-1], rhos[-1], rhos[0]], aspect='auto')
    axs[0, 1].set_title("Hough Accumulator")
    axs[0, 1].set_xlabel("Theta (degrees)")
    axs[0, 1].set_ylabel("Rho (pixels)")

    # Plot 3: Detected Line Points in Hough Space
    axs[1, 0].imshow(accumulator, cmap='hot', extent=[thetas[0], thetas[-1], rhos[-1], rhos[0]], aspect='auto')
    axs[1, 0].scatter([thetas[idx[1]] for idx in detected_lines], 
                      [rhos[idx[0]] for idx in detected_lines], 
                      color='cyan', marker='o', label="Detected Lines")
    axs[1, 0].set_title("Detected Lines in Hough Space")
    axs[1, 0].set_xlabel("Theta (degrees)")
    axs[1, 0].set_ylabel("Rho (pixels)")
    axs[1, 0].legend()

    # Plot 4: Original Image with Detected Lines
    axs[1, 1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Detected Lines on Image")
    axs[1, 1].axis("off")

   
    plt.tight_layout()

    # Save figure to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)

    # Convert buffer to image
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def detect_lines(image=None, num_rho=180, num_theta=180, blur_ksize=5, low_threshold=50, high_threshold=150, hough_threshold_ratio=0.6):
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
        edged_image: The image after Canny edge detection.
        accumulator: The Hough Transform accumulator.
        rhos: Array of rho values.
        thetas: Array of theta values.
        output_image: The image with detected lines drawn.
    '''
    
    edged_image = detect_edges_cv(image, low_threshold=low_threshold, high_threshold=high_threshold)

    height, width = edged_image.shape[:2]
    half_height, half_width = height / 2, width / 2

    diagonal = int(math.sqrt(height**2 + width**2))
    dtheta = 180 / num_theta
    drho = (2 * diagonal) / num_rho

    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-diagonal, diagonal, step=drho)
    
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    for y in range(height):
        for x in range(width):
            if edged_image[y, x] != 0:
                center = [y - half_height, x - half_width]
                for theta_idx in range(len(thetas)):
                    rho = (center[1] * cos_thetas[theta_idx]) + (center[0] * sin_thetas[theta_idx])
                    rho_idx = np.argmin(abs(rhos - rho))
                    accumulator[rho_idx][theta_idx] += 1

    max_val = np.max(accumulator)
    threshold = max_val * hough_threshold_ratio
    
    detected_lines = [(y, x) for y in range(accumulator.shape[0]) for x in range(accumulator.shape[1]) if accumulator[y, x] > threshold]
    
    filtered_lines = lines_nms(detected_lines, accumulator, rhos, thetas)
    

    output_image = image.copy()
    for rho, theta in filtered_lines:
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + half_width
        y0 = (b * rho) + half_height
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        output_image = cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return visualize_lines(edged_image, accumulator, detected_lines, filtered_lines,  rhos, thetas, output_image)
    
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

    theta = np.deg2rad(np.arange(0, 360, 5))  
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for y, x in edge_pixels:
        for r in range(min_radius, max_radius):
            a = (x - r * cos_t).astype(int)  
            b = (y - r * sin_t).astype(int)  
            
            valid_idx = (0 <= a) & (a < w) & (0 <= b) & (b < h)
            np.add.at(accumulator, (b[valid_idx], a[valid_idx], r - min_radius), 1)
    return accumulator

def non_maximum_suppression(circles, accumulator, min_dist=40, r_min = 10):
    """Applies Non-Maximum Suppression to remove overlapping circles based on Euclidean distance."""
    filtered_circles = []
    
    sorted_circles = sorted(circles, key=lambda c: -accumulator[c[1], c[0], c[2] - r_min])
    
    for x, y, r in sorted_circles:
        keep = True
        for x2, y2, r2 in filtered_circles:
            dist = np.linalg.norm(np.array((x, y)) - np.array((x2, y2)))  
            if dist < min_dist + abs(r - r2):  
                keep = False
                break  
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
    return non_maximum_suppression(circles, accumulator, min_dist=min_dist, r_min= min_radius)

def draw_detected_circles(image, detected_circles):
    """Draws detected circles on the original image."""
    output_image = image.copy()
    for a, b, r in detected_circles:
        cv2.circle(output_image, (a, b), r, (0, 255, 0), 2)
    return output_image


def visualize_results(gray_image, edges, accumulator, detected_circles):
    """
    Generates a 2x2 visualization plot and returns the image as a NumPy array.

    Args:
        gray_image (np.ndarray): Original grayscale image.
        edges (np.ndarray): Edge-detected image.
        accumulator (np.ndarray): Hough accumulator array.
        detected_circles (list): List of detected circles (x, y, r).

    Returns:
        np.ndarray: Image representation of the plotted figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    axs[0, 0].imshow(gray_image, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(edges, cmap='gray')
    axs[0, 1].set_title("Edge Detection")
    axs[0, 1].axis("off")

    accumulator_projection = np.sum(accumulator, axis=2)  # Sum over radius
    axs[1, 0].imshow(accumulator_projection, cmap='hot', extent=[0, gray_image.shape[1], gray_image.shape[0], 0])
    axs[1, 0].set_title("Hough Accumulator (Summed Over Radii)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(gray_image, cmap='gray')
    axs[1, 1].set_title("Detected Circles")
    axs[1, 1].axis("off")

    for x, y, r in detected_circles:
        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
        axs[1, 1].add_patch(circle)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)

    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def hough_circle_detection(image, **kwargs):
    """Performs full Hough circle detection."""
    edge_param = {k: v for k, v in kwargs.items() if k in ['low_threshold', 'high_threshold', 'max_edge_val', 'min_edge_val']}
    r_min = kwargs.get('r_min', 10)
    r_max = kwargs.get('r_max', 50)

    radius_range=(r_min, r_max)
    edges = detect_edges_cv(image, **edge_param)
    accumulator = initialize_hough_circle_space(edges, r_min, r_max)
    print('accumlator done!!')
    accumulator = compute_hough_circle_votes(edges, accumulator, radius_range)
    detected_circles = extract_circle_peaks(accumulator, radius_range, kwargs.get('bin_threshold', 0.5))
    result = visualize_results(image, edges, accumulator, detected_circles)
    return result

def draw_ellipses(image, ellipses):
    """
    Draws detected ellipses on the input image.

    Parameters:
    - image: Original grayscale image
    - ellipses: List of detected ellipses with (center, major_axis_pair, minor_axis_pair)
    """
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for (center, (p1_major, p2_major), (p1_minor, p2_minor)) in ellipses:
        xc, yc = center

        major_length = np.linalg.norm(np.array(p1_major) - np.array(p2_major))
        minor_length = np.linalg.norm(np.array(p1_minor) - np.array(p2_minor))

        a = max(1, int(major_length / 2))  
        b = max(1, int(minor_length / 2))  

        angle = np.arctan2(p2_major[1] - p1_major[1], p2_major[0] - p1_major[0])* 180.0 / np.pi
        cv2.ellipse(output, (yc, xc), (a, b), 90-angle, 0, 360, (255, 0, 0), 1)
        cv2.line(output, (p1_major[1], p1_major[0]), (p2_major[1], p2_major[0]), (255, 0, 0), 1)  
        cv2.line(output, (p1_minor[1], p1_minor[0]), (p2_minor[1], p2_minor[0]), (0, 0, 255), 1)  
        cv2.circle(output, (yc, xc), 3, (0, 255, 0), -1)

    return output

def plot_hough_ellipse_steps(original, edges, accumulator, focus_pairs, detected_ellipses):
    """
    Plots all the steps of the Hough Transform ellipse detection process.
    
    Parameters:
    - original: Original grayscale image
    - edges: Edge-detected binary image
    - accumulator: Dictionary storing votes for (center, focus pairs)
    - focus_pairs: List of focus point pairs considered
    - detected_ellipses: List of detected ellipses
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Step 1: Original Image
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")
    
    # Step 2: Edge Detection Output
    axs[0, 1].imshow(edges, cmap='gray')
    axs[0, 1].set_title("Edge Detection")
    axs[0, 1].axis("off")
    

    # Step 4: Accumulator Heatmap
    accumulator_image = np.zeros_like(original, dtype=np.float32)
    for center, pairs in accumulator.items():
        accumulator_image[center] += len(pairs)
    
    axs[1, 0].imshow(accumulator_image, cmap='hot')
    axs[1, 0].set_title("Accumulator (Ellipse Centers)")
    axs[1, 0].axis("off")
    
    # Step 5: Final Ellipse Detections
    result_image = draw_ellipses(original, detected_ellipses)
    axs[1, 1].imshow(result_image)
    axs[1, 1].set_title("Detected Ellipses")
    axs[1, 1].axis("off")
    

    plt.tight_layout()

    # Save figure to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)

    # Convert buffer to image
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def compute_hough_ellipse_focus(edges, min_d=10, max_d=200, step_size=10):
    """
    Hough Transform for ellipse detection using the focus-based representation.
    
    Parameters:
    - edges: Binary edge image (numpy array)
    - min_d, max_d: Range of sum of distances defining the ellipse
    - step_size: Step size for reducing the search space

    Returns:
    - accumulator: Dictionary storing votes for (f1, f2, d)
    - focus_pairs: List of focus point pairs considered
    """
    edge_points = np.argwhere(edges > 0)
    accumulator = defaultdict(list)
    focus_pairs = []
    for (x1, y1), (x2, y2) in combinations(edge_points, 2):
        f1, f2 = (x1, y1), (x2, y2)
        focus_pairs.append((f1, f2))
        center = ((x1 + x2)//2 , (y1 + y2)//2)
        distance = np.linalg.norm(np.array(f1) - np.array(f2))
        if min_d<distance  and distance <max_d:
            accumulator[center].append(((x1, y1), (x2, y2))) 

    return accumulator, focus_pairs


def extract_best_ellipses(accumulator, threshold_factor=0.5, tolerance=20):
    """
    Extracts best ellipses based on list length filtering and ensures major and minor axes are nearly perpendicular.

    Parameters:
    - accumulator: Dictionary mapping center points to lists of focus point pairs
    - threshold_factor: Fraction of max list length to filter weak detections
    - tolerance: Angle tolerance (in degrees) for perpendicularity check

    Returns:
    - List of (center, max_pair, min_pair) for detected ellipses
    """
    if not accumulator:
        return []
    max_length = max(len(pairs) for pairs in accumulator.values())
    threshold_length = threshold_factor * max_length
    best_ellipses = []

    for center, pairs in accumulator.items():
        if len(pairs) >= threshold_length:
            distances = [(p1, p2, np.linalg.norm(np.array(p1) - np.array(p2))) for p1, p2 in pairs]

            max_pair = max(distances, key=lambda x: x[2])  # (p1, p2, max_distance)
            min_pair = min(distances, key=lambda x: x[2])  # (p1, p2, min_distance)

            major_vector = np.array(max_pair[1]) - np.array(max_pair[0])
            minor_vector = np.array(min_pair[1]) - np.array(min_pair[0])

            dot_product = np.dot(major_vector, minor_vector)
            angle = np.degrees(np.arccos(dot_product / (np.linalg.norm(major_vector) * np.linalg.norm(minor_vector))))

            if 90 - tolerance <= angle <= 90 + tolerance:
                best_ellipses.append((center, max_pair[:2], min_pair[:2]))  # Only take point pairs, not distances

    return best_ellipses

def hough_ellipse_detection(image, min_d=10, max_d=50, step_size=10, threshold_factor=0.5, **kwargs):
    """
    Performs Hough Transform for ellipse detection and extracts the best ellipses.

    Args:
        edges (np.ndarray): Binary edge-detected image.
        min_d (int): Minimum major axis length.
        max_d (int): Maximum major axis length.
        step_size (int): Step size for iterating over possible major axes.
        threshold_factor (float): Fraction of max votes to consider as a threshold.

    Returns:
        list: List of detected ellipses (x1, y1, x2, y2, d).
    """
    edge_param = {k: v for k, v in kwargs.items() if k in ['low_threshold', 'high_threshold', 'max_edge_val', 'min_edge_val']}
    edges = detect_edges_cv(image, **edge_param)
    accumulator , focus_pairs= compute_hough_ellipse_focus(edges, min_d, max_d, step_size)
    best_ellipses = extract_best_ellipses(accumulator, threshold_factor=threshold_factor)
    
    return plot_hough_ellipse_steps(image,edges,accumulator,focus_pairs, best_ellipses)

# plot_hough_ellipse_steps(image,edges,accumulator,focus_pairs, best_ellipses)
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