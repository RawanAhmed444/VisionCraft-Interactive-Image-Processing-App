import numpy as np

def ideal_filter(dft_shifted, cutoff = 10, type = "lp"):
    rows, cols = dft_shifted.shape
    center = (rows // 2, cols // 2)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center[0])**2 + (j - center[1])**2) <= cutoff:
                if type == "lp":
                    mask[i, j] = 1
                elif type == "hp":
                    mask[i, j] = 0
                    
    return mask

def calculate_dft(img):
    dft = np.fft.fft2(img)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shifted) + 1e-10) 
    return dft_shifted, magnitude_spectrum

def filter_image(dft_shifted, mask):
    filtered_dft = dft_shifted*mask
    inverse_dft = np.fft.ifftshift(filtered_dft)
    image_filtered = np.fft.ifft2(inverse_dft)
    image_filtered = np.abs(image_filtered)
    image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)
    return image_filtered


    