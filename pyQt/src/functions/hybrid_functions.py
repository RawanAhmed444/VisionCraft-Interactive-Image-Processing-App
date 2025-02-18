from frequency_functions import ideal_filter
import numpy as np

def hybrid_filter(img1, img2, cutoff1 = 10, cutoff2 = 10, type1 = "lp", type2 = "lp"):
    
    image1 = ideal_filter(img1, cutoff1, type1)
    image2 = ideal_filter(img2, cutoff2, type2)
    
    hybird_image = image1 + image2
    
    return np.clip(hybird_image, 0, 255).astype(np.uint8)