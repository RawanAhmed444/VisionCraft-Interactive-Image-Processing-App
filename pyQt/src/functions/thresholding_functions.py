import numpy as np


def threshold_image(img, T = 128, local = False, kernal= 4, k = 2):
    if local:
        binary_img = localthresholding(img, kernal, k)
    else: 
        if T is None:
            T = np.mean(img)
        binary_img = globalthresholding(img, T)
    
    return binary_img

def globalthresholding(image, T = 128):
    binary_img  = (image > T).astype(np.uint8) * 255
    return binary_img


def localthresholding(image, kernal = 4, k = 2):
    """
    Local thresholding using Niblack's method.
    """
    #use odd kernal size
    if kernal % 2 == 0:
        kernal += 1
    #handling borders
    pad = kernal // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0) #may be needed to implemented on my own
    
    binary_img = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighbor = padded_image[i:i+kernal, j:j+kernal]
            mean = np.mean(neighbor)
            std = np.std(neighbor)    
            
            T = mean + k * std
            if image[i,j] > T :
                binary_img[i, j] = 255
    return binary_img