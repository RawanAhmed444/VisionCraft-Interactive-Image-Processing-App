'''
The Active Contour Model (ACM), commonly known as Snakes, is an edge-detection technique used in image processing.
It is an iterative algorithm that deforms a curve (or contour) towards object boundaries based on energy minimization.

The goal is to minimize the total energy (internal - external - constrain)
'''

import numpy as np 
from functions.noise_functions import apply_gaussian_filter
from utils import convert_to_grayscale


#contour funcitons
import numpy as np
from scipy.ndimage import gaussian_filter

def initialize_snake(center, radius, points=100):
    """Create a circular snake around a given center with a specific radius."""
    s = np.linspace(0, 2 * np.pi, points)
    x = center[0] + radius * np.cos(s)
    y = center[1] + radius * np.sin(s)
    return np.array([x, y]).T

def internal_energy_matrix(n_points, alpha=0.1, beta=0.1, gamma=0.1):
    """Create the internal energy matrix A for smoothness and elasticity."""
    A = np.zeros((n_points, n_points))
    for i in range(n_points):
        A[i, i] = 2 * alpha + 6 * beta
        A[i, (i - 1) % n_points] = -alpha - 4 * beta
        A[i, (i + 1) % n_points] = -alpha - 4 * beta
        A[i, (i - 2) % n_points] = beta
        A[i, (i + 2) % n_points] = beta
    return np.linalg.inv(A + gamma * np.eye(n_points))

def external_energy(image, sigma=1.0):
    """Compute the edge energy from the image gradient."""
    smoothed_image = gaussian_filter(image, sigma)
    gy, gx = np.gradient(smoothed_image)
    edge_energy = np.sqrt(gx**2 + gy**2)
    # Normalize gradients
    gx = gx / np.max(np.abs(gx))
    gy = gy / np.max(np.abs(gy))
    return edge_energy, gx, gy

def optimize_snake_step(image, snake, inv_matrix, gx, gy, gamma=0.1, w_edge=1.0):
    """
    Performs a single iteration of the active contour optimization.
    
    :param image: Input image.
    :param snake: Current snake contour (Nx2 array).
    :param inv_matrix: Precomputed internal energy inverse matrix.
    :param gx: X-gradient of external energy.
    :param gy: Y-gradient of external energy.
    :param gamma: Step size.
    :param w_edge: Edge force weight.
    
    :return: Updated snake contour (Nx2 array).
    """
    # Interpolate external forces at snake points
    int_x = np.clip(snake[:, 0].astype(int), 0, image.shape[1] - 1)
    int_y = np.clip(snake[:, 1].astype(int), 0, image.shape[0] - 1)

    fx = gx[int_y, int_x]
    fy = gy[int_y, int_x]

    # Normalize external forces
    force_magnitude = np.sqrt(fx**2 + fy**2)
    force_magnitude[force_magnitude == 0] = 1  # Avoid division by zero
    fx /= force_magnitude
    fy /= force_magnitude

    # External force vector
    force = np.stack([fx, fy], axis=1) * w_edge
    print("force",force.shape)

    # Update snake using internal and external forces
    new_snake = np.dot(inv_matrix, snake + gamma * force)

    # Ensure snake points stay within image boundaries
    new_snake[:, 0] = np.clip(new_snake[:, 0], 0, image.shape[1] - 1)
    new_snake[:, 1] = np.clip(new_snake[:, 1], 0, image.shape[0] - 1)

    return new_snake
