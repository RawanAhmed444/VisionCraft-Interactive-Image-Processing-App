from functions.frequency_functions import ideal_filter, calculate_dft, filter_image
import numpy as np
import cv2
def hybrid_filter(img1, img2, cutoff1=10, cutoff2=10, type1="lp", type2="hp"):
    # Calculate DFT for both images
    dft_shifted1, _ = calculate_dft(img1)
    dft_shifted2, _ = calculate_dft(img2)
    
    # Create masks
    mask1 = ideal_filter(dft_shifted1, cutoff1, type1)
    mask2 = ideal_filter(dft_shifted2, cutoff2, type2)
    
    # Filter images
    filtered_img1 = filter_image(dft_shifted1, mask1)
    filtered_img2 = filter_image(dft_shifted2, mask2)
    
    # Resize images if necessary
    if filtered_img1.shape != filtered_img2.shape:
        filtered_img1 = cv2.resize(filtered_img1, (filtered_img2.shape[1], filtered_img2.shape[0]))
    
    # Combine images
    hybrid_image = filtered_img1 + filtered_img2
    
    return np.clip(hybrid_image, 0, 255).astype(np.uint8)
