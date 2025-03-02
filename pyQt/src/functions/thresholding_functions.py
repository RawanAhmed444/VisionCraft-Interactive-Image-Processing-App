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


def localthresholding(image, kernal=10, k=0.5):
    """
    Local thresholding using Niblack's method.
    """
    # Use odd kernel size
    if kernal % 2 == 0:
        kernal += 1

    # Handling borders
    pad = kernal // 2
    padded_image = np.pad(image, pad, mode='constant')  # Zero-padding

    binary_img = np.zeros_like(image, dtype=np.uint8)

    # Apply local thresholding
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the local neighborhood
            neighbor = padded_image[i:i + kernal, j:j + kernal]

            # Debug: Print the shape of the neighbor
            if neighbor.shape != (kernal, kernal):
                print(f"Warning: Invalid neighbor shape at ({i}, {j}): {neighbor.shape}")
                continue  # Skip this pixel if the neighbor is invalid

            # Calculate mean and standard deviation of the neighborhood
            mean = np.mean(neighbor, axis=(0, 1))
            std = np.std(neighbor, axis=(0, 1))

            # Calculate the threshold using Niblack's method
            T = mean + k * std

            # Debug: Print the threshold and pixel value
            print(f"Pixel ({i}, {j}): T = {T}, image[i,j] = {image[i, j]}")

            # Apply the threshold
            if image[i, j] > T:
                binary_img[i, j] = 255

    return binary_img
