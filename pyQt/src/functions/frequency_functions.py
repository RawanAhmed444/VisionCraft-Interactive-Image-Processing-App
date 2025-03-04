import numpy as np

def ideal_filter(dft_shifted, cutoff=10, type="low_pass"):
    rows, cols = dft_shifted.shape
    center = (rows // 2, cols // 2)
    
    # Create a grid of distances from the center
    x = np.arange(cols) - center[1]
    y = np.arange(rows) - center[0]
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx**2 + yy**2)
    
    # Create the mask based on the filter type
    if type == "low_pass" or "lp":  # Low-pass filter
        mask = (distance <= cutoff).astype(np.uint8)
    elif type == "high_pass" or "hp":  # High-pass filter
        mask = (distance > cutoff).astype(np.uint8)
    else:
        raise ValueError("Invalid filter type. Use 'lp' or 'hp'.")
    
    return mask

def calculate_dft(img):
    dft = np.fft.fft2(img)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted, np.abs(dft_shifted)

def filter_image(dft_shifted, mask):
    filtered_dft = dft_shifted * mask
    inverse_dft = np.fft.ifftshift(filtered_dft)
    image_filtered = np.fft.ifft2(inverse_dft)
    image_filtered = np.abs(image_filtered)
    image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)
    return image_filtered


    