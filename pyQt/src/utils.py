import numpy as np
import cv2

def convert_to_grayscale(image):
    """
    Converts a BGR image to grayscale using the weighted sum method.

    :param image: Input color image (BGR format).
    :return: Grayscale image as NumPy array.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    
    # Apply grayscale conversion formula
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    return gray_image.astype(np.uint8)

def magnitude(x, y):
    return np.sqrt(x**2 + y**2)

def compute_gradient(image, axis = 0):
    """
    NOT IMPLEMENTED
    Computes the gradient of an image using the Sobel operator.
    """
    """
    Computes the gradient of a 2D array along the specified axis.
    
    For interior points, uses central differences.
    For boundaries, uses forward difference at the beginning and backward difference at the end.
    
    :param image: 2D NumPy array (e.g., grayscale image).
    :param axis: Axis along which to compute the gradient (0 for rows, 1 for columns).
    :return: Gradient of the image along the specified axis.
    """
    if image.ndim != 2:
        raise ValueError("my_gradient currently supports only 2D arrays.")
    
    grad = np.zeros_like(image, dtype=np.uint8)
    
    if axis == 0:
        # Compute gradient along rows
        # Forward difference for the first row
        grad[0, :] = image[1, :] - image[0, :]
        # Central differences for interior rows
        for i in range(1, image.shape[0] - 1):
            grad[i, :] = (image[i+1, :] - image[i-1, :]) / 2.0
        # Backward difference for the last row
        grad[-1, :] = image[-1, :] - image[-2, :]
        
    elif axis == 1:
        # Compute gradient along columns
        # Forward difference for the first column
        grad[:, 0] = image[:, 1] - image[:, 0]
        # Central differences for interior columns
        for j in range(1, image.shape[1] - 1):
            grad[:, j] = (image[:, j+1] - image[:, j-1]) / 2.0
        # Backward difference for the last column
        grad[:, -1] = image[:, -1] - image[:, -2]
        
    else:
        raise ValueError("Unsupported axis. Use axis=0 for rows or axis=1 for columns.")
        
    return np.clip(grad, 0, 255).astype(np.uint8)
def convolve(img, kernel):
    """
    Applies convolution to an image using a given kernel (without using convolve2d).

    :param img: Input grayscale image (NumPy array).
    :param kernel: Filter kernel (NumPy array).
    :return: Convolved image.
    """
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2  # Padding for kernel centering
    
    # Pad the image (edge mode to preserve boundary information)
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    # Create an empty output image
    output = np.zeros_like(img, dtype=np.float32)
    
    # Perform convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = img_padded[i:i+kernel_height, j:j+kernel_width]  # Extract region
            output[i, j] = np.sum(region * kernel)  # Apply filter
    
    return np.clip(output, 0, 255).astype(np.uint8)  # Normalize output to valid range

def gaussian_kernel(size, sigma=1.0):
    """
    Generates a Gaussian kernel.

    :param size: Kernel size (odd number).
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: Normalized Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)  # Normalize kernel to sum to 1

def cumsum(arr):
    """
    Computes the cumulative sum of an array.

    :param arr: List of numbers.
    :return: List containing the cumulative sum.
    """
    cumulative_sum = [0] * len(arr)  # Initialize output array with zeros
    
    cumulative_sum[0] = arr[0]  # First element remains the same

    for i in range(1, len(arr)):  # Start from the second element
        cumulative_sum[i] = cumulative_sum[i - 1] + arr[i]  # Add previous sum

    return np.cumsum(arr)