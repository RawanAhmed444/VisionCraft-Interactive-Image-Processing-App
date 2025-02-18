import numpy as np

def add_uniform_noise(img, intensity=50):
    """
    Adds uniform noise to an image using only NumPy.
    
    :param img: Input grayscale image (NumPy array).
    :param intensity: Maximum noise level (default=50).
    :return: Noisy image.
    """
    noise = np.random.uniform(-intensity, intensity, img.shape)  # Generate noise
    noisy_img = img.astype(np.int16) + noise  # Convert to int16 to avoid overflow
    noisy_img = np.clip(noisy_img, 0, 255)  # Ensure pixel values remain in range
    return noisy_img.astype(np.uint8)  # Convert back to uint8

def add_gaussian_noise(img, mean=0, std=25):
    """
    Adds Gaussian noise to an image using only NumPy.
    
    :param img: Input grayscale image (NumPy array).
    :param mean: Mean of the Gaussian noise (default=0).
    :param std: Standard deviation of the noise (default=25).
    :return: Noisy image.
    """
    noise = np.random.normal(mean, std, img.shape)  # Generate Gaussian noise
    noisy_img = img.astype(np.int16) + noise  # Convert to int16 to prevent overflow
    noisy_img = np.clip(noisy_img, 0, 255)  # Ensure pixel values remain in range
    return noisy_img.astype(np.uint8)  # Convert back to uint8

def add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02):
    """
    Adds salt & pepper noise to an image using only NumPy.
    
    :param img: Input grayscale image (NumPy array).
    :param salt_prob: Probability of salt noise (default=0.02).
    :param pepper_prob: Probability of pepper noise (default=0.02).
    :return: Noisy image.
    """
    noisy_img = np.copy(img)  # Copy original image
    
    # Generate random mask for salt (white) noise
    salt_mask = np.random.rand(*img.shape) < salt_prob
    noisy_img[salt_mask] = 255  # Assign maximum intensity
    
    # Generate random mask for pepper (black) noise
    pepper_mask = np.random.rand(*img.shape) < pepper_prob
    noisy_img[pepper_mask] = 0  # Assign minimum intensity
    
    return noisy_img
