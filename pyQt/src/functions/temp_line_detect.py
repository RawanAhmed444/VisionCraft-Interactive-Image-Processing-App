import cv2
import numpy as np
import math

def temp_detect_lines(image, num_rho=180, num_theta=180, blur_ksize=5, low_threshold=50, high_threshold=150, hough_threshold_ratio=0.6):
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