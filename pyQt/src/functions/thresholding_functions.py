import numpy as np


def threshold_image(img, T = 128, local = False, kernal= 4, k = 2):
    if local:
        binary_img = localthresholding(img, kernal, k)
    else: 
        if T is None:
            T = np.mean(img)
        binary_img = globalthresholding(img, T)
    
    return binary_img

def globalthresholding(image, T = 128, value = 255):
    binary_img  = (image > T).astype(np.uint8) * value
    return binary_img
def integral_image(image):
    """
    Compute the integral image of a given image.
    
    :param image: Input grayscale image (2D NumPy array).
    :return: Integral image.
    """
    return np.cumsum(np.cumsum(image, axis=0), axis=1)

def _mean_std(image, w):
    """
    Return local mean and standard deviation of each pixel using a
    neighborhood defined by a rectangular window with size w x w.
    The algorithm uses integral images to speed up computation.

    :param image: Input grayscale image (2D NumPy array).
    :param w: Odd window size (e.g., 3, 5, 7, ..., 21, ...).
    :return: Tuple (m, s) where:
             - m: 2D array of local mean values.
             - s: 2D array of local standard deviation values.
    """
    if w == 1 or w % 2 == 0:
        raise ValueError(f"Window size w = {w} must be odd and greater than 1.")

    # Pad the image to handle borders
    pad = w // 2
    padded = np.pad(image.astype('float'), ((pad, pad), (pad, pad)), mode='reflect')
    padded_sq = padded * padded

    # Compute integral images
    integral = integral_image(padded)
    integral_sq = integral_image(padded_sq)

    # Define the kernel for computing sums
    kern = np.zeros((w + 1, w + 1))
    kern[0, 0] = 1
    kern[0, -1] = -1
    kern[-1, 0] = -1
    kern[-1, -1] = 1

    # Compute local sums using the kernel
    sum_full = np.zeros_like(image, dtype=np.float64)
    sum_sq_full = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Coordinates in the padded image
            i_pad = i + pad
            j_pad = j + pad

            # Compute the sum using the integral image
            sum_full[i, j] = (
                integral[i_pad + 1, j_pad + 1] -
                integral[i_pad + 1, j_pad - w] -
                integral[i_pad - w, j_pad + 1] +
                integral[i_pad - w, j_pad - w]
            )
            sum_sq_full[i, j] = (
                integral_sq[i_pad + 1, j_pad + 1] -
                integral_sq[i_pad + 1, j_pad - w] -
                integral_sq[i_pad - w, j_pad + 1] +
                integral_sq[i_pad - w, j_pad - w]
            )

    # Compute local mean and standard deviation
    m = sum_full / (w ** 2)
    g2 = sum_sq_full / (w ** 2)
    s = np.sqrt(np.maximum(g2 - m * m, 0))  # Ensure non-negative values

    return m, s

def localthresholding(image, Kernal_size=3, k=0.2):
    """
    Applies Niblack local thresholding to an image.

    :param image: Input grayscale image (2D NumPy array).
    :param window_size: Odd size of pixel neighborhood window (e.g., 3, 5, 7...).
    :param k: Value of parameter k in the threshold formula.
    :return: Binary image after thresholding.
    """
    # Compute local mean and standard deviation
    m, s = _mean_std(image, Kernal_size)

    # Calculate the threshold using Niblack's method
    threshold = m - k * s

    # Apply the threshold
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)

    return binary_image

