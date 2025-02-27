'''
The Active Contour Model (ACM), commonly known as Snakes, is an edge-detection technique used in image processing.
It is an iterative algorithm that deforms a curve (or contour) towards object boundaries based on energy minimization.

The goal is to minimize the total energy (internal - external - constrain)
'''

import numpy as np 
from functions.noise_functions import apply_gaussian_filter

#contour funcitons

def init_contour(image_dim = [254,254],points=100, radius= 50, center=None, shape = "circle"):
    if center is None:
        center = (shape[0]//2 , shape[1]//2)
    
    angles = np.linspace(0, 2* np.pi, points)
    
    x  = center[0] + radius * np.cos(angles)
    y  = center[1] + radius * np.sin(angles)
    
    return np.array([x, y]).T  # Shape: (num_points, 2)

def evolve_contour(image, initial_contour, alpha, beta, gamma, iterations = 100):
    
    pass

def compute_edge_energy(image):
    smooth_image = apply_gaussian_filter(image)
    dI_dx = np.gradient(smooth_image, axis=1)
    dI_dy = np.gradient(smooth_image, axis=0)
    
    energy = np.sqrt(dI_dx**2 + dI_dy**2)
    
    return -energy.astype(np.uint8) 


# Convertion functions
def contour_to_chain(contour):
    
    pass

def compute_perimeter(contour):
    
    pass

def compute_area(contour):
    
    pass


