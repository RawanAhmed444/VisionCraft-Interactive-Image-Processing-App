import numpy as np
from functions.frequency_functions import ideal_filter, calculate_dft, filter_image
from functions.hybrid_functions import hybrid_filter
from utils import convert_to_grayscale
class FrequencyProcessor:
    def __init__(self): 
        self.image = None   
        self.dft_shifted = None
        self.magnitude_spectrum = None
        self.filtered_image = None
        self.mask = None
        self.filter_type = None

        
    def create_hybrid_image(self,img1 , img2, cutoff1=10, cutoff2=10, type1="lp", type2="lp"):
        img1 = convert_to_grayscale(img1)
        img2 = convert_to_grayscale(img2)
        self.hybrid_image = hybrid_filter(img1, img2, cutoff1, cutoff2, type1, type2)
        return self.hybrid_image
   
    def set_image(self, image):
        """Sets the image and computes its DFT"""
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image input. Expected a NumPy array.")
        if len(image.shape) != 2:
            self.image = convert_to_grayscale(image)
        self.dft_shifted, self.magnitude_spectrum = calculate_dft(self.image)


    def get_magnitude_spectrum(self):
        """Returns the magnitude spectrum of the image"""
        if self.magnitude_spectrum is None:
            raise ValueError("No magnitude spectrum available. Please call set_image() first.")
        return self.magnitude_spectrum

    def get_dft(self):
        """Returns the shifted DFT of the image"""
        if self.dft_shifted is None:
            raise ValueError("No DFT available. Please call set_image() first.")
        return self.dft_shifted
    
    def get_filtered_image(self):
        """Returns the filtered image"""
        if self.filtered_image is None:
            raise ValueError("No filtered image available. Please apply a filter first.")
        return self.filtered_image

    def apply_filter(self, radius=10, filter_type=None):
        """Applies a frequency-domain filter (low-pass or high-pass)"""
        if self.dft_shifted is None:
            raise ValueError("No image set. Please call set_image() first.")
        self.filter_type = filter_type
        print(self.filter_type)
        self.mask = ideal_filter(dft_shifted=self.dft_shifted,cutoff=radius, type=filter_type)
        filtered_image = filter_image(self.dft_shifted, self.mask)
        self.filtered_image = filtered_image
        return filtered_image
