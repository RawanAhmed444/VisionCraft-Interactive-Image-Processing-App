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

def convolve(img, kernel):
    """
    Applies convolution to an image using a given kernel with zero-order interpolation.
    
    :param img: Input grayscale image (NumPy array).
    :param kernel: Filter kernel (NumPy array).
    :return: Convolved image.
    """
    height, width = img.shape
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2

    output = np.zeros_like(img, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            sum_val = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    # Calculate source pixel coordinates
                    y = i + ki - pad_h
                    x = j + kj - pad_w
                    
                    # Apply zero-order interpolation by clamping coordinates
                    y = max(0, min(y, height - 1))
                    x = max(0, min(x, width - 1))
                    
                    sum_val += img[y, x] * kernel[ki, kj]
            
            output[i, j] = sum_val

    return np.clip(output, 0, 255).astype(np.uint8)

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