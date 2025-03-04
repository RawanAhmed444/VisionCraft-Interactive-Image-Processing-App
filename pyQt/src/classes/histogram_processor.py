import numpy as np
import matplotlib.pyplot as plt
from utils import convert_to_grayscale
from functions.histogram_functions import (
    compute_histogram, compute_cdf, normalize, apply_histogram_equalization,
    draw_histo_and_distribution_curve, draw_image_histogram_and_distribution
)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget
from PyQt5.QtGui import QImage, QPixmap


class HistogramProcessor:
    """Handles histogram computation, CDF calculation, normalization, equalization, and visualization for grayscale and color images."""

    def __init__(self):
        self.image = None
        self.histograms = None
        self.cdfs = None
        self.cdf_normalized = None
        self.equalized_image = None
        self.is_color = False

    def set_image(self, image):
        """
        Sets the image and computes histograms for grayscale or each RGB channel.

        :param image: Input grayscale or color image.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image input. Expected a NumPy array.")
        
        self.image = image
        self.is_color = (len(image.shape) > 2) # Check if the image has 3 channels
        if self.is_color:
            self.histograms = {color: compute_histogram(image[:, :, idx]) 
                               for idx, color in enumerate(["Red", "Green", "Blue"])}
            self.cdfs = {color: compute_cdf(hist) for color, hist in self.histograms.items()}
            self.cdf_normalized = {color: normalize(cdf) for color, cdf in self.cdfs.items()}
        else:
            self.histograms = compute_histogram(image)
            self.cdfs = compute_cdf(self.histograms)
            self.cdf_normalized = normalize(self.cdfs)
        self.apply_histogram_equalization()


    def transform_to_grayscale(self):
        """
        Converts a color image to grayscale using a weighted sum method.
        
        :return: Grayscale image.
        """
        if not self.is_color:
            raise ValueError("Image is already grayscale.")
        
        grayscale_image = convert_to_grayscale(self.image)
        return grayscale_image

    def apply_histogram_equalization(self):
        """
        Applies histogram equalization using the normalized CDF.

        :return: Equalized grayscale or color image.
        """
        if self.image is None:
            raise ValueError("Image not set. Please call set_image() first.")

        if self.is_color:
            # Apply equalization separately for R, G, B channels
            equalized_channels = [self.cdf_normalized[color][self.image[:, :, idx]] 
                                  for idx, color in enumerate(["Red", "Green", "Blue"])]
            self.equalized_image = np.stack(equalized_channels, axis=-1)
        else:
            self.equalized_image = apply_histogram_equalization(self.image, self.cdf_normalized)
        
        return self.equalized_image

    def plot_histogram_with_distribution(self, channel="Grayscale"):
        """
        Plots histogram and its distribution function (CDF) for a specific channel.

        :param channel: 'Grayscale' or 'Red', 'Green', 'Blue'.
        """
        if self.histograms is None:
            raise ValueError("No histogram available. Set an image first.")
        
        if self.is_color and channel not in ["Red", "Green", "Blue"]:
            raise ValueError("Invalid channel. Choose 'Red', 'Green', or 'Blue'.")

        # Select histogram and CDF based on the channel
        if self.is_color:
            hist = self.histograms[channel]
            cdf = self.cdfs[channel]
            color = channel.lower()
        else:
            hist = self.histograms
            cdf = self.cdfs
            color = "black"

        # Plot histogram
        fig, ax1 = plt.subplots()
        ax1.bar(range(256), hist, color=color, alpha=0.6, label=f'{channel} Histogram')
        ax1.set_xlabel("Pixel Intensity")
        ax1.set_ylabel("Frequency", color=color)
        
        # Plot CDF on the same graph
        ax2 = ax1.twinx()
        ax2.plot(range(256), cdf / cdf.max(), color="blue", label="CDF")  # Normalize CDF
        ax2.set_ylabel("Cumulative Distribution", color="blue")

        plt.title(f"{channel} Histogram & Distribution")
        plt.legend()
        plt.show()

    def plot_all_histograms(self):
        """
        Plots histograms and CDFs for all channels (Grayscale or RGB).
        """
        if self.is_color:
            for channel in ["Red", "Green", "Blue"]:
                self.plot_histogram_with_distribution(channel)
        else:
            self.plot_histogram_with_distribution()

    def plot_histogram_with_normal_distribution(self, channel="Grayscale"):
        """
        Plots histogram with an overlaid normal distribution curve for a specific channel.

        :param channel: 'Grayscale' or 'Red', 'Green', 'Blue'.
        """
        if self.histograms is None:
            raise ValueError("No histogram available. Set an image first.")
        
        if self.is_color and channel not in ["Red", "Green", "Blue"]:
            raise ValueError("Invalid channel. Choose 'Red', 'Green', or 'Blue'.")

        # Select data based on the channel
        if self.is_color:
            idx = ["Red", "Green", "Blue"].index(channel)
            data = self.image[:, :, idx].flatten()
        else:
            data = self.image.flatten()

        # Plot histogram with normal distribution curve
        draw_histo_and_distribution_curve(data, bins=256)

    def plot_all_histograms_with_normal_distribution(self):
        """
        Plots histograms with normal distribution curves for all channels (Grayscale or RGB).
        """
        if self.is_color:
            for channel in ["Red", "Green", "Blue"]:
                self.plot_histogram_with_normal_distribution(channel)
        else:
            self.plot_histogram_with_normal_distribution()
            

class HistogramVisualizationWidget(QWidget):
    """A custom QWidget to display histogram-related visualizations."""

    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Histogram Visualizations")
        self.setGeometry(100, 100, 1200, 800)

        # Create a tab widget to organize visualizations
        tab_widget = QTabWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(tab_widget)

        # Tab 1: Original and Equalized Images
        image_tab = QWidget()
        image_layout = QHBoxLayout(image_tab)

        # Display original image
        original_image_label = QLabel()
        original_image_label.setPixmap(self._convert_image_to_pixmap(self.processor.image))
        image_layout.addWidget(original_image_label)

        # Display equalized image
        equalized_image_label = QLabel()
        equalized_image_label.setPixmap(self._convert_image_to_pixmap(self.processor.equalized_image))
        image_layout.addWidget(equalized_image_label)

        tab_widget.addTab(image_tab, "Images")

        # Tab 2: Histograms and CDFs
        histogram_tab = QWidget()
        histogram_layout = QVBoxLayout(histogram_tab)

        # Plot histograms and CDFs
        if self.processor.is_color:
            for channel in ["Red", "Green", "Blue"]:
                fig = self._plot_histogram_and_cdf(channel)
                canvas = FigureCanvas(fig)
                histogram_layout.addWidget(canvas)
        else:
            fig = self._plot_histogram_and_cdf()
            canvas = FigureCanvas(fig)
            histogram_layout.addWidget(canvas)

        tab_widget.addTab(histogram_tab, "Histograms & CDFs")

        # Tab 3: Histograms with Normal Distribution Curves
        distribution_tab = QWidget()
        distribution_layout = QVBoxLayout(distribution_tab)

        # Plot histograms with normal distribution curves
        if self.processor.is_color:
            for channel in ["Red", "Green", "Blue"]:
                fig = self._plot_histogram_with_normal_distribution(channel)
                canvas = FigureCanvas(fig)
                distribution_layout.addWidget(canvas)
        else:
            fig = self._plot_histogram_with_normal_distribution()
            canvas = FigureCanvas(fig)
            distribution_layout.addWidget(canvas)

        tab_widget.addTab(distribution_tab, "Histograms & Distributions")

    def _convert_image_to_pixmap(self, image):
        """
        Converts a NumPy image array to a QPixmap.

        :param image: Input image as a NumPy array.
        :return: QPixmap representation of the image.
        """
        if self.processor.is_color:
            # Convert BGR to RGB for display
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            height, width = image.shape
            qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)

        return QPixmap.fromImage(qimage)

    def _plot_histogram_and_cdf(self, channel="Grayscale"):
        """
        Plots histogram and CDF for a specific channel.

        :param channel: 'Grayscale' or 'Red', 'Green', 'Blue'.
        :return: Matplotlib figure.
        """
        if self.processor.is_color and channel not in ["Red", "Green", "Blue"]:
            raise ValueError("Invalid channel. Choose 'Red', 'Green', or 'Blue'.")

        # Select histogram and CDF based on the channel
        if self.processor.is_color:
            hist = self.processor.histograms[channel]
            cdf = self.processor.cdfs[channel]
            color = channel.lower()
        else:
            hist = self.processor.histograms
            cdf = self.processor.cdfs
            color = "black"

        # Create figure
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.bar(range(256), hist, color=color, alpha=0.6, label=f'{channel} Histogram')
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency", color=color)

        # Plot CDF on the same graph
        ax2 = ax.twinx()
        ax2.plot(range(256), cdf / cdf.max(), color="blue", label="CDF")  # Normalize CDF
        ax2.set_ylabel("Cumulative Distribution", color="blue")

        plt.title(f"{channel} Histogram & Distribution")
        plt.legend()
        return fig

    def _plot_histogram_with_normal_distribution(self, channel="Grayscale"):
        """
        Plots histogram with an overlaid normal distribution curve for a specific channel.

        :param channel: 'Grayscale' or 'Red', 'Green', 'Blue'.
        :return: Matplotlib figure.
        """
        if self.processor.is_color and channel not in ["Red", "Green", "Blue"]:
            raise ValueError("Invalid channel. Choose 'Red', 'Green', or 'Blue'.")

        # Select data based on the channel
        if self.processor.is_color:
            idx = ["Red", "Green", "Blue"].index(channel)
            data = self.processor.image[:, :, idx].flatten()
        else:
            data = self.processor.image.flatten()

        # Create figure
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.hist(data, bins=256, density=True, edgecolor='black', alpha=0.7, label='Histogram')

        # Compute normal distribution curve
        mean, std = np.mean(data), np.std(data)
        x = np.linspace(min(data), max(data), 100)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax.plot(x, y, color='red', label='Normal Distribution')

        ax.set_title(f"{channel} Histogram with Distribution Curve")
        ax.set_xlabel("Values")
        ax.set_ylabel("Density")
        ax.legend()
        return fig