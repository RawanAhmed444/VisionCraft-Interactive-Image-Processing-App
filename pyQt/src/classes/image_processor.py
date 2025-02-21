import cv2
import numpy as np
from functions.normalize_and_equalize import equalize_grayscale_image, equalize_color_image, normalize_grayscale_image, normalize_color_image
import numpy as np
class ImageProcessor:
    def __init__(self, image):
        if isinstance(image, np.ndarray):
            self.image = image
        else:
            raise ValueError("Input image must be a NumPy array")
        self.gray_scale = len(self.image.shape) == 2
        self.equlized_image = None
        self.normalized_image = None
        
    def set_image(self, image):
        self.image = image

        
    def equalize(self):
        if self.gray_scale:
            self.equlized_image = equalize_grayscale_image(self.image)
        else:
            self.equlized_image = equalize_color_image(self.image)


    def normalize(self):
        if self.gray_scale:
            self.normalized_image = normalize_grayscale_image(self.image)
        else:
            self.normalized_image = normalize_color_image(self.image)

    def get_equalized_image(self):
        if self.equlized_image is None:
            self.equalize()
        return self.equlized_image
    
    def get_normalized_image(self):
        if self.normalized_image is None:
            self.normalize()    
        return self.normalized_image