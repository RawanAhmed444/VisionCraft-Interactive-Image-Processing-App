import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import cv2


def draw_histo_and_distribution_curve(data, bins):
    """
    Plots a histogram with an overlaid normal distribution curve.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot histogram
    plt.hist(data, bins=bins, density=True, edgecolor='black', alpha=0.7, label='Histogram')
    
    # Compute normal distribution curve
    mean, std = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 100)
    y = norm.pdf(x, mean, std)
    
    # Overlay normal distribution curve
    plt.plot(x, y, color='red', label='Normal Distribution')
    
    # Add labels and title
    plt.title('Histogram with Distribution Curve')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    
    # Show plot
    plt.show()
    
    return x, y

def draw_image_histogram_and_distribution(image, bins=256):
    """
    takes an image and plots its histogram with an overlaid normal distribution curve.
    
    Returns:
    x (numpy array): X values for the normal distribution curve.
    y (numpy array): Y values for the normal distribution curve.
    """
    flattened_data = image.flatten()
    return draw_histo_and_distribution_curve(flattened_data, bins=bins)


def color_to_grayscale_and_histograms(image):
   
    #Convert BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the image into R, G, B channels
    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

    # Apply the grayscale conversion formula
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b

    # Convert the result to uint8 (0-255 range)
    gray_image = gray_image.astype(np.uint8)

    colors = {'Red': r, 'Green': g, 'Blue': b}
    plt.figure(figsize=(12, 6))
    
    for idx, (color_name, channel) in enumerate(colors.items(), 1):
        # Compute histogram
        hist, bin = np.histogram(channel.flatten(), 256, [0, 256])
        
        # Compute cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf_normalized = cdf / float(cdf.max())  # Normalize to [0, 1]
        
        # Plot histogram and CDF
        plt.subplot(2, 3, idx)
        # plt.hist(channel.flatten(), bins=bin, color=color_name.lower(), alpha=0.6, label=f'{color_name} Histogram')
        plt.bar(bin[:-1], hist, color=color_name.lower(), alpha=0.6, label=f'{color_name} Histogram')
        plt.legend()
        
        plt.subplot(2, 3, idx + 3)
        plt.plot(cdf_normalized, color=color_name.lower(), label=f'{color_name} CDF')
        plt.legend()

    plt.tight_layout()
    plt.show()

    return gray_image




