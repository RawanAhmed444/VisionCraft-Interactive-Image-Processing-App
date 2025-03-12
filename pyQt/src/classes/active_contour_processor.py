import cv2
import numpy as np
from functions.active_contour_functions import (
    initialize_snake,
    external_energy,
    gradient_descent_step
)
from utils import convert_to_grayscale, get_dimensions

class ActiveContourProcessor:
    """
    A class to perform active contour segmentation.
    It uses modular functions from active_contour_functions for actual processing.
    """
    def __init__(self):
        """
        Initializes the ActiveContourProcessor with an input image.
        """
        self.image = None  # Input image (grayscale)
        self.snake = None  # Final snake contour
        self.edge_maps = {}  # Dictionary to store intermediate results

    def set_image(self, image):
        """
        Sets the input image for active contour segmentation.
        
        :param image: Input image in BGR or grayscale format.
        """
        # Convert to grayscale if necessary
        self.image = convert_to_grayscale(image)

    def detect_contour(self, center=None, radius=None, alpha=0.1, beta=0.1, gamma=0.1,
                       w_edge=1.0, sigma=1.0, iterations=250, convergence=0.01, points=100):
        """
        Detects the contour using the active contour algorithm.
        
        :param center: Tuple (x, y) for the initial snake center. If None, the image center is used.
        :param radius: Initial radius of the snake. If None, a default value is used.
        :param alpha: Elasticity weight.
        :param beta: Curvature weight.
        :param gamma: Step size.
        :param w_edge: Edge force weight.
        :param sigma: Gaussian smoothing sigma.
        :param iterations: Maximum number of iterations.
        :param convergence: Convergence threshold.
        :param points: Number of points in the snake.
        :return: Final snake contour.
        """
        if self.image is None:
            raise ValueError("No image set. Use set_image() to provide an input image.")

        # Set default center and radius if not provided
        if center is None:
            center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        if radius is None:
            radius = min(self.image.shape[0], self.image.shape[1]) // 4

        # Compute external energy (image gradient)
        edge_energy, gx, gy = external_energy(self.image, sigma)

        # Initialize snake
        snake = initialize_snake(center, radius, points)

        # Optimize snake
        for iter_num in range(iterations):
            new_snake = gradient_descent_step(snake, self.image, gx, gy, alpha, beta, gamma, w_edge)

            # Check for convergence
            displacement = np.mean(np.sqrt(np.sum((new_snake - snake) ** 2, axis=1)))
            if displacement < convergence:
                print(f"Convergence reached at iteration {iter_num}.")
                break

            snake = new_snake

        self.snake = snake
        return snake

    def get_contour(self):
        """
        Retrieves the final snake contour.
        
        :return: Final snake contour if computed, otherwise None.
        """
        return self.snake

    def visualize_contour(self):
        """
        Visualizes the final snake contour over the input image.
        """
        if self.image is None or self.snake is None:
            raise ValueError("No image or contour available. Run detect_contour() first.")

        plt.figure(figsize=(8, 8))
        plt.imshow(self.image, cmap='gray')
        plt.plot(self.snake[:, 0], self.snake[:, 1], 'r-', linewidth=2, label="Final Contour")
        plt.title("Active Contour Segmentation")
        plt.legend()
        plt.xticks([]), plt.yticks([])  # Hide axis ticks
        plt.show()