import numpy as np
from functions.frequency_functions import ideal_filter, calculate_dft, filter_image
class FrequencyProcessor:
    def __init__(self): 
        self.image = None   
        self.dft_shifted = None
        self.magnitude_spectrum = None
        self.filtered_image = None

    def apply_filter(self, cutoff = 10, type = "hp"):
        mask = ideal_filter(self.dft_shifted, cutoff, type)
        self.filtered_image = filter_image(self.dft_shifted, mask)
        return self.filtered_image 

    def set_image(self, image):
        self.image = image
        self.dft_shifted, self.magnitude_spectrum = calculate_dft(image)
        
    def get_magnitude_spectrum(self):
        return self.magnitude_spectrum

    def get_dft(self):
        return self.dft_shifted