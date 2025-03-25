import cv2
import numpy as np
from functions.active_contour_functions import (
    initialize_snake,
    external_energy,
    internal_energy_matrix,
    optimize_snake_step
)
from utils import convert_to_grayscale, get_dimensions
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
        self.history = []  # Stores the contour evolution history

    def set_image(self, image):
        """
        Sets the input image for active contour segmentation.
        
        :param image: Input image in BGR or grayscale format.
        """
        # Convert to grayscale if necessary
        self.image = convert_to_grayscale(image)

    def detect_contour(self, center=None, radius=None, alpha=0.5, beta=0.7, gamma=1,
                       w_edge=10, sigma=1.4, iterations=1000, convergence=0.0, points=100):
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
        # Initialize snake and internal energy matrix
        snake = initialize_snake(center, radius, points)
        print("snake",snake.shape)
        inv_matrix = internal_energy_matrix(len(snake), alpha, beta, gamma)
        print("inv_matrix",inv_matrix.shape)
        # Reset history before running
        self.history = []

       # Iteratively optimize snake
        for iter_num in range(iterations):
            # Store current state for visualization
            self.history.append({
                "iteration": iter_num,
                "snake": snake.copy(),
                "internal_energy": inv_matrix.copy(),
                "external_energy": edge_energy.copy()
            })
            print(f"histry {len(self.history)}")
            
            new_snake = optimize_snake_step(self.image,snake,inv_matrix , gx, gy, gamma, w_edge)

            # Check for convergence
            displacement = np.mean(np.sqrt(np.sum((new_snake - snake) ** 2, axis=1)))
            if displacement < convergence:
                print(f"Convergence reached at iteration {iter_num}.")
                break

            snake = new_snake

        self.snake = snake
        return snake, self.history

    def get_contour(self):
        """
        Retrieves the final snake contour.
        
        :return: Final snake contour if computed, otherwise None.
        """
        return self.snake



    def visualize_contour(self):
        if self.image is None or not self.history:
            raise ValueError("No image or history available. Run detect_contour() first.")

        fig, ax_img = plt.subplots(figsize=(7, 5))
        plt.subplots_adjust(bottom=0.25)  # More space for slider

        ax_img.imshow(self.image, cmap='gray')
        contour_line, = ax_img.plot([], [], 'r-', linewidth=2, label="Snake Contour")
        ax_img.set_title("Active Contour Result")
        ax_img.legend()

        def update_plot(iteration):
            iteration = int(iteration)
            data = self.history[iteration]

            contour_line.set_data(data["snake"][:, 0], data["snake"][:, 1])
            ax_img.set_title(f"Iteration {iteration}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()  # Ensure real-time updates

        # **Move slider lower and make it bigger**
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04], facecolor='lightgray')
        slider = Slider(ax_slider, "Iteration", 0, len(self.history) - 1, valinit=0, valstep=1)
        slider.poly.set_linewidth(50)

        # **Fix dragging issue by properly linking update function**
        slider.on_changed(lambda val: update_plot(slider.val))  # Explicitly use slider.val

        update_plot(0)  # Initialize

        plt.show(block=True)  # Ensure the slider is interactive