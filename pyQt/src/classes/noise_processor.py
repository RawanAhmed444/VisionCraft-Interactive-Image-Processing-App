import numpy as np
from functions.noise_functions import add_uniform_noise, add_gaussian_noise, add_salt_pepper_noise, apply_average_filter, apply_gaussian_filter, apply_median_filter
from utils import convert_to_grayscale
class NoiseProcessor:
    """Handles noise addition and filtering for grayscale images."""
    
    def __init__(self):
        self.image = None
        self.noisy_image = None
        self.filtered_images = {}

    def set_image(self, image):
        """
        Sets the input image.

        :param image: Grayscale image (NumPy array).
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image input. Expected a NumPy array.")
        self.image = image
        self.noisy_image = image

    def add_noise(self, noise_type="uniform", **kwargs):
        """
        Adds noise to the image.

        :param noise_type: Type of noise ('uniform', 'gaussian', 'salt_pepper').
        :param kwargs: Additional parameters for the noise functions.
        :return: Noisy image.
        """
        if self.image is None:
            raise ValueError("No image set. Use set_image() first.")

        if noise_type == "uniform":
            self.noisy_image = add_uniform_noise(self.image, **kwargs)
        elif noise_type == "gaussian":
            self.noisy_image = add_gaussian_noise(self.image, **kwargs)
        elif noise_type == "salt_pepper":
            self.noisy_image = add_salt_pepper_noise(self.image, **kwargs)
        else:
            raise ValueError("Invalid noise type. Choose 'uniform', 'gaussian', or 'salt_pepper'.")

        return self.noisy_image

    def apply_filters(self, **kwargs): 
        """
        Applies average, Gaussian, and median filters to the noisy image.

        :return: Dictionary containing filtered images.
        """
        if self.noisy_image is None:
            
            raise ValueError("No noisy image available. Apply noise first.")
        
        self.noisy_image =   convert_to_grayscale(self.noisy_image)     
        self.filtered_images = {
            "average": apply_average_filter(self.noisy_image, **kwargs), # kernel_size=3
            "gaussian": apply_gaussian_filter(self.noisy_image, **kwargs), # kernel_size=3, sigma=1.0
            "median": apply_median_filter(self.noisy_image, **kwargs) # kernel_size=3
        }
        return self.filtered_images

    def get_noisy_image(self):
        """Returns the noisy image."""
        if self.noisy_image is None:
            raise ValueError("No noisy image available. Apply noise first.")
        return self.noisy_image

    def get_filtered_images(self):
        """Returns the filtered images."""
        if not self.filtered_images:
            raise ValueError("No filters applied. Call apply_filters() first.")
        return self.filtered_images
