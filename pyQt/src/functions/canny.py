
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import convert_to_grayscale, gaussian_kernel, convolve
from functions.edge_functions import roberts_gradients, prewitt_gradients
def apply_gaussian_filter(image, kernel_size=5, sigma=1.4):
    """Applies a Gaussian filter to the image."""
    if image is None:
        raise ValueError("Input image is None.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def detect_edges_cv(circles_image_gray, low_threshold=50, high_threshold=150):
    return cv2.Canny(circles_image_gray, low_threshold, high_threshold)

def magnitude(Gx, Gy):
    """Computes the gradient magnitude."""
    return np.hypot(Gx, Gy)

def non_maximum_suppression(G, theta):
    height, width = G.shape
    nms = np.zeros_like(G, dtype=np.uint8)
    theta = theta % 180

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle = theta[y, x]
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                p1, p2 = G[y, x-1], G[y, x+1]
            elif 22.5 <= angle < 67.5:
                p1, p2 = G[y-1, x+1], G[y+1, x-1]
            elif 67.5 <= angle < 112.5:
                p1, p2 = G[y-1, x], G[y+1, x]
            elif 112.5 <= angle < 157.5:
                p1, p2 = G[y-1, x-1], G[y+1, x+1]

            if G[y, x] >= p1 and G[y, x] >= p2:
                nms[y, x] = G[y, x]

    return nms

def apply_double_thresholding(image, low_threshold, high_threshold, max_edge_val=255, min_edge_val=20):
    strong_edges = np.zeros_like(image)
    weak_edges = np.zeros_like(image)

    strong_edges[image >= high_threshold] = max_edge_val
    weak_edges[(image >= low_threshold) & (image < high_threshold)] = min_edge_val
    return strong_edges, weak_edges

def apply_hysteresis(strong_edges, weak_edges, max_edge_val=255):
    height, width = strong_edges.shape
    final_edges = np.copy(strong_edges)
    stack = list(zip(*np.where(strong_edges == max_edge_val)))

    while stack:
        y, x = stack.pop()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and weak_edges[ny, nx] != 0:
                    final_edges[ny, nx] = max_edge_val
                    weak_edges[ny, nx] = 0
                    stack.append((ny, nx))

    return final_edges

def compute_sobel_gradients(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx = convolve(image, kernel_x)
    Gy = convolve(image, kernel_y)
    return Gx, Gy

def canny_edge_detection(image, low_threshold=None, high_threshold=None, max_edge_val=255, min_edge_val=0):
    gray = convert_to_grayscale(image)
    smooth = apply_gaussian_filter(gray)
    Gx, Gy = roberts_gradients(image)
    G = magnitude(Gx, Gy)
    G_normalized = np.uint8(G / G.max() * 255)

    theta = np.rad2deg(np.arctan2(Gy, Gx)) % 180
    nms = non_maximum_suppression(G_normalized, theta)

    max_val = np.max(nms)
    low_threshold = low_threshold if low_threshold is not None else max_val * 0.1
    high_threshold = high_threshold if high_threshold is not None else max_val * 0.5

    strong_edges, weak_edges = apply_double_thresholding(nms, low_threshold, high_threshold, max_edge_val, min_edge_val)
    final_edges = apply_hysteresis(strong_edges, weak_edges)

    return  final_edges
