import cv2
import numpy as np
import math
from collections import defaultdict

def detect_lines(image, num_rho=180, num_theta=180, blur_ksize=5, low_threshold=50, high_threshold=150, hough_threshold_ratio=0.6):
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
    edged_image = cv2.Canny(blured_image, low_threshold, high_threshold)

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
        edged_image = cv2.Canny(blured_image, min_edge_threshold, max_edge_threshold)

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