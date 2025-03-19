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
    """
    Create a circular snake around a given center with a specific radius.
    
    :param center: Tuple (x, y) for the center of the snake.
    :param radius: Radius of the snake.
    :param points: Number of points in the snake.
    :return: Array of snake points.
    """
    s = np.linspace(0, 2 * np.pi, points)
    x = center[0] + radius * np.cos(s)
    y = center[1] + radius * np.sin(s)
    return np.array([x, y]).T

def external_energy(image, sigma=1.0):
    """
    Compute the external energy from the image gradient.
    
    :param image: Input image.
    :param sigma: Gaussian smoothing sigma.
    :return: Edge energy, gradient in x, gradient in y.
    """
    if len(image.shape) == 3:  
        image = convert_to_grayscale(image)
    
    smoothed_image = gaussian_filter(image, sigma)
    gy, gx = np.gradient(smoothed_image)
    edge_energy = np.sqrt(gx**2 + gy**2)
    return edge_energy, gx, gy

def gradient_descent_step(snake, image, gx, gy, alpha=0.1, beta=0.1, gamma=0.1, w_edge=1.0):
    """
    Perform a single gradient descent step to update the snake.
    
    :param snake: Current snake points.
    :param image: Input image.
    :param gx: Gradient in x-direction.
    :param gy: Gradient in y-direction.
    :param alpha: Elasticity weight.
    :param beta: Curvature weight.
    :param gamma: Step size.
    :param w_edge: Edge force weight.
    :return: Updated snake points.
    """
    n = len(snake)
    new_snake = np.zeros_like(snake)

    for i in range(n):
        # Internal forces (smoothness and elasticity)
        prev = snake[(i - 1) % n]
        curr = snake[i]
        next_ = snake[(i + 1) % n]
        internal_force = alpha * (prev - 2 * curr + next_) + beta * (prev - 2 * curr + next_)

        # External forces (image gradient)
        x, y = curr.astype(int)
        x = np.clip(x, 0, image.shape[1] - 1)
        y = np.clip(y, 0, image.shape[0] - 1)
        external_force = w_edge * np.array([gx[y, x], gy[y, x]])

        # Update snake point
        new_snake[i] = curr + gamma * (internal_force + external_force)

    # Clip snake points to image boundaries
    new_snake[:, 0] = np.clip(new_snake[:, 0], 0, image.shape[1] - 1)
    new_snake[:, 1] = np.clip(new_snake[:, 1], 0, image.shape[0] - 1)

    return new_snake