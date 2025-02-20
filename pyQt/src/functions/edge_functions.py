import numpy as np 
from utils import convolve, magnitude, convert_to_grayscale
from thresholding_functions import globalthresholding
from noise_functions import apply_gaussian_filter
import cv2 
import matplotlib.pyplot as plt 



def smooth_image(image, kernel_size=5, sigma=1.4):
    """Smooths the image using a Gaussian filter."""
    return apply_gaussian_filter(image, kernel_size, sigma)

def compute_sobel_gradients(image):
    """
    Computes the Sobel gradients of an image.
    
    :param image: Grayscale image.
    :return: Tuple (Gx, Gy) representing horizontal and vertical gradients.
    """
    # Define Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    
    Gx = convolve(image, kernel_x)
    Gy = convolve(image, kernel_y)
    return Gx, Gy

def non_maximum_suppression(G, theta):
    """
    Suppresses non-maximum pixels in the gradient magnitude image.
    
    :param G: Gradient magnitude image.
    :param theta: Gradient direction in degrees.
    :return: Image after non-maximum suppression.
    """
    height, width = G.shape
    nms = np.zeros_like(G)
    
    # Loop through image (ignore boundaries for simplicity)
    for y in range(1, height-1):
        for x in range(1, width-1):
            angle = theta[y, x]
            # Determine neighbors along the gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = G[y, x-1], G[y, x+1]
            elif (22.5 <= angle < 67.5):
                q, r = G[y-1, x+1], G[y+1, x-1]
            elif (67.5 <= angle < 112.5):
                q, r = G[y-1, x], G[y+1, x]
            elif (112.5 <= angle < 157.5):
                q, r = G[y-1, x-1], G[y+1, x+1]
            
            # Retain pixel if it's a local maximum
            if G[y, x] >= q and G[y, x] >= r:
                nms[y, x] = G[y, x]
            else:
                nms[y, x] = 0
    return nms


def apply_double_thresholding(G, low_threshold, high_threshold, strong_edge=255, weak_edge=50):
    """
    Applies double thresholding to classify pixels as strong, weak, or non-edges.

    :param G: Non-maximum suppressed gradient magnitude image.
    :param low_threshold: Lower threshold value.
    :param high_threshold: Higher threshold value.
    :return: Tuple (strong_edges, weak_edges) with binary masks.
    """
    # Initialize masks for strong and weak edges
    strong_edges = np.zeros_like(G, dtype=np.uint8)
    weak_edges = np.zeros_like(G, dtype=np.uint8)

    # Define edge strength values
    STRONG = strong_edge
    WEAK = weak_edge  

    # Classify pixels
    strong_edges[G >= high_threshold] = STRONG
    weak_edges[(G >= low_threshold) & (G < high_threshold)] = WEAK

    return strong_edges, weak_edges

def apply_hysteresis(strong_edges, weak_edges):
    """
    Performs edge tracking by hysteresis to retain only connected weak edges.

    :param strong_edges: Binary mask of strong edges.
    :param weak_edges: Binary mask of weak edges.
    :return: Final edge-detected image.
    """
    height, width = strong_edges.shape
    final_edges = strong_edges.copy()

    # Directions for 8-connected neighbors
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    # Iterate over all weak edge pixels
    for y in range(1, height-1):
        for x in range(1, width-1):
            if weak_edges[y, x] > 0:  # If it's a weak edge
                for dy, dx in directions:
                    if strong_edges[y + dy, x + dx] == 255:  # Connected to a strong edge
                        final_edges[y, x] = 255
                        break  # Stop checking neighbors if connected

    return final_edges




def sobel_edge_detection(image):
    """
    Detects edges using the Sobel operator.
    
    :param image: Input BGR or grayscale image.
    :return: Sobel edge map (normalized to 0-255).
    """
    gray = convert_to_grayscale(image)
    # Optional Gaussian smoothing before gradient calculation:
    blurred = smooth_image(gray, kernel_size=3, sigma=1.0)
    Gx, Gy = compute_sobel_gradients(blurred)
    G = magnitude(Gx, Gy)
    return np.uint8(255 * (G / np.max(G)))


def prewitt_edge_detection(image):
    """
    Detects edges using the Prewitt operator.
    
    :param image: Input BGR or grayscale image.
    :return: Prewitt edge map.
    """
    gray = convert_to_grayscale(image)
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]])
    Gx = convolve(gray, kernel_x)
    Gy = convolve(gray, kernel_y)
    
    Gx = np.float32(Gx)
    Gy = np.float32(Gy)
    
    G = np.uint8(magnitude(Gx, Gy))
    G = globalthresholding(G, 50, 255)
    return G
    
    
def roberts_edge_detection(image):
    """
    Detects edges using the Roberts operator.
    
    :param image: Input BGR or grayscale image.
    :return: Roberts edge map.
    """
    gray = convert_to_grayscale(image)
    kernel_x = np.array([[1, 0],
                         [0, -1]])
    kernel_y = np.array([[0, 1],
                         [-1, 0]])
    Gx = convolve(gray, kernel_x)
    Gy = convolve(gray, kernel_y)
    return np.int8(magnitude(Gx, Gy))

def canny_edge_detection(image, low_threshold=None, high_threshold=None, max_edge_val=255, min_edge_val=0):
    """
    Full Canny edge detection pipeline (from scratch).
    
    :param image: Input BGR or grayscale image.
    :param low_threshold: Lower threshold for double thresholding.
    :param high_threshold: Higher threshold for double thresholding.
    :return: Final edge map.
    """
    gray = convert_to_grayscale(image)
    
    smooth = smooth_image(gray, kernel_size=5, sigma=1.4)
    
    Gx, Gy = compute_sobel_gradients(smooth)
    G = magnitude(Gx, Gy)
    
    theta = np.arctan2(Gy, Gx) * (180 / np.pi)
    theta[theta < 0] += 180
    
    nms = non_maximum_suppression(G, theta)
    
    max_val = np.max(nms)
    low_threshold = low_threshold if low_threshold is not None else max_val * 0.1
    high_threshold = high_threshold if high_threshold is not None else max_val * 0.5
    strong_edges, weak_edges = apply_double_thresholding(nms, low_threshold, high_threshold, max_edge_val, min_edge_val)
    
    final_edges = apply_hysteresis(strong_edges, weak_edges)
    return np.int8(final_edges)

# Canny Edge Detection
def canny_edge_detection(img, low_threshold=None, high_threshold=None):
    # Conversion of image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Noise reduction step 
    img = apply_gaussian_filter(img, 5, 1.4)
    
	# # Calculating the gradients 
    # gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
    # gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # Define Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) 

    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]) 

    # Calculating the gradients 
    Gx = convolve(img, kernel_x)
    Gy = convolve(img, kernel_y)

	# # Conversion of Cartesian coordinates to polar coordinates
    # mag, ang = cv2.cartToPolar(Gx, Gy, angleInDegrees = True)

    # Manual Calculation of Magnitude and Angle 
    mag = magnitude(Gx, Gy)
    ang = np.arctan2(Gy, Gx) * 180 / np.pi
    
	# Setting the minimum and maximum thresholds for double thresholding
    max_mag = np.max(mag)
    if not low_threshold:
        low_threshold = max_mag * 0.1
    if not high_threshold:
        high_threshold = max_mag * 0.5
        
	# Get the dimentions of the input image
    height, width = img.shape
    
	# Loop though every pixel in the grayscale image
    for x_i in range(width):
        for y_i in range(height):
            grad_ang = ang[y_i, x_i]
            grad_ang = abs(grad_ang-180) if abs(grad_ang) > 180 else abs(grad_ang)
            
			# Select the neighbours of the target pixel according to the gradient direction
            
			# In the x direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = x_i - 1, y_i 
                neighb_2_x, neighb_2_y = x_i + 1, y_i 
                
			# In the top right (diagonal-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = x_i - 1, y_i - 1 
                neighb_2_x, neighb_2_y = x_i + 1, y_i + 1
                
			# In the y direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = x_i , y_i - 1 
                neighb_2_x, neighb_2_y = x_i , y_i + 1
                
			# In the top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = x_i - 1, y_i + 1 
                neighb_2_x, neighb_2_y = x_i + 1, y_i - 1
                
			# Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = x_i - 1, y_i  
                neighb_2_x, neighb_2_y = x_i + 1, y_i 
                
			# Non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[y_i, x_i] < mag[neighb_1_y, neighb_1_x]:
                    mag[y_i, x_i] = 0
                    continue
            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[y_i, x_i] < mag[neighb_2_y, neighb_2_x]:
                    mag[y_i, x_i] = 0
                    

	# Hysteresis Thresholding (NOT IMPLEMENTED YET)
    weak_ids = np.zeros_like(img) 
    strong_ids = np.zeros_like(img)               
    ids = np.zeros_like(img) 
		
	# double thresholding step 
    for i_x in range(width): 
        for i_y in range(height): 
              
            grad_mag = mag[i_y, i_x] 
              
            if grad_mag < low_threshold: 
                mag[i_y, i_x]= 0
            elif high_threshold > grad_mag >= low_threshold: 
                ids[i_y, i_x]= 1
            else: 
                ids[i_y, i_x]= 2
                
    return mag

# Prewitt Edge Detection
# def prewitt_edge_detection(image):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply horizontal Prewitt kernel
#     kernel_x = np.array([[-1, 0, 1],
#                          [-1, 0, 1],
#                          [-1, 0, 1]])
#     horizontal_edges = convolve(gray_image, kernel_x)
    
#     # Apply vertical Prewitt kernel
#     kernel_y = np.array([[-1, -1, -1],
#                          [0, 0, 0],
#                          [1, 1, 1]])
#     vertical_edges = convolve(gray_image, kernel_y)

#     # Ensure both arrays have the same data type

    
#     # Compute gradient magnitude
#     gradient_magnitude = magnitude(horizontal_edges, vertical_edges)
    
#     # Optional: Apply thresholding to highlight edges
#     thresh_value, edges = globalthresholding(gradient_magnitude, 50, 255)
    
#     return edges

# # Roberts Edge Detection
# def roberts_edge_detection(image):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     roberts_cross_v = np.array( [[1, 0 ], 
#                                 [0,-1 ]]) 

#     roberts_cross_h = np.array( [[ 0, 1 ], 
#                                 [ -1, 0 ]]) 
    
#     # Apply the filter (convolution)
#     vertical = convolve(gray_image, roberts_cross_v ) 
#     horizontal = convolve(gray_image, roberts_cross_h ) 

#     # Compute gradient magnitude
#     gradient_mag = magnitude(horizontal, vertical)

#     return gradient_mag

def display_edge_detection(image_path, edge_detection_func, title):
    # Read the input image
    image = cv2.imread(image_path)
    
    # Apply edge detection
    edges = edge_detection_func(image)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Detected Edges
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f'{title} Edge Detection')
    plt.axis('off')
    
    plt.show()

# # Usage examples:
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', sobel_edge_detection, 'Sobel')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', canny_edge_detection, 'Canny')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', prewitt_edge_detection, 'Prewitt')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', roberts_edge_detection, 'Roberts')
