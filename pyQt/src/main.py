import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize

from processor_factory import ProcessorFactory
from classes.histogram_processor import HistogramVisualizationWidget
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QFrame,QTabWidget,QSpacerItem,QSizePolicy,
    QVBoxLayout, QWidget, QScrollArea,QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QCheckBox, QGroupBox, QGridLayout,
    QSlider
)

from processor_factory import ProcessorFactory
from functions.hough_transform_functions import detect_lines,detect_circles
from functions.active_contour_functions import initialize_snake, external_energy

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class ActiveContourVisualizationWidget(QWidget):
    """A separate window to visualize Active Contour evolution and energy maps."""

    def __init__(self,parent, processor = None):
        super().__init__(parent)
        self.processor = processor
        self.history = []
        self.current_index = 0
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Active Contour Visualization")
        self.setGeometry(100, 100, 1200, 800)

        # Create a tab widget to organize visualizations
        self.tab_widget = QTabWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.tab_widget)

        # Add tabs for different visualizations
        self.original_image_label = QLabel()
        self.internal_energy_label = QLabel()
        self.external_energy_label = QLabel()
        self.contour_evolution_label = QLabel()

        self.add_tab(self.original_image_label, "Original Image")
        self.add_tab(self.internal_energy_label, "Internal Energy")
        self.add_tab(self.external_energy_label, "External Energy")
        self.add_tab(self.contour_evolution_label, "Contour Evolution")

        # Slider to navigate through iterations
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_visualization)

        layout.addWidget(self.slider)

    def add_tab(self, widget, title):
        """Adds a new tab with the given widget and title."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(widget)
        self.tab_widget.addTab(tab, title)

    def set_history(self, history):
        """
        Receives contour evolution history from the processor and updates visualization.
        """
        self.history = history
        print(len(history))
        self.slider.setMaximum(len(history) - 1)
        self.update_visualization()

    def update_visualization(self):
        """Updates the displayed images based on the current iteration index."""
        if not self.history:
            return

        idx = self.slider.value()
        data = self.history[idx]
        print(data)

        # Convert images to QPixmap format and update QLabel widgets
        self.original_image_label.setPixmap(self.numpy_to_pixmap(self.processor.image))
        self.internal_energy_label.setPixmap(self.numpy_to_pixmap(data["internal_energy"], cmap="hot"))
        self.external_energy_label.setPixmap(self.numpy_to_pixmap(data["external_energy"], cmap="cool"))
        self.contour_evolution_label.setPixmap(self.draw_contour(data["snake"]))

    def numpy_to_pixmap(self, array, cmap="gray"):

        if len(array.shape) == 2:  # Grayscale image
            array = (255 * (array - array.min()) / (array.max() - array.min())).astype(np.uint8)
            height, width = array.shape
            qimage = QImage(array.data, width, height, QImage.Format_Grayscale8)
            return QPixmap.fromImage(qimage)
        return QPixmap()

    def draw_contour(self, snake):
        """Draws the active contour over the original image."""
        img_copy = cv2.cvtColor(self.processor.image.copy(), cv2.COLOR_GRAY2BGR)

        # Draw contour in red
        for point in snake:
            x, y = int(point[0]), int(point[1])
            cv2.circle(img_copy, (x, y), 1, (0, 0, 255), -1)

        return self.numpy_to_pixmap(cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY))





class ActiveContourTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        active_contour_frame = QFrame()
        active_contour_frame.setObjectName("active_contour_frame")
        active_contour_layout = QVBoxLayout(active_contour_frame)
        active_contour_layout.setAlignment(Qt.AlignTop)

        self.centerX = QSpinBox()
        self.centerX.setRange(0, 1000)
        self.centerX.setValue(250)

        self.centerY = QSpinBox()
        self.centerY.setRange(0, 1000)
        self.centerY.setValue(250)

        self.radius = QSpinBox()
        self.radius.setRange(1, 500)
        self.radius.setValue(50)

        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 10.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(0.1)

        self.beta = QDoubleSpinBox()
        self.beta.setRange(0.0, 10.0)
        self.beta.setSingleStep(0.1)
        self.beta.setValue(0.1)

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.0, 10.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(0.1)

        self.iterations = QSpinBox()
        self.iterations.setRange(1, 1000)
        self.iterations.setValue(100)

        self.points = QSpinBox()
        self.points.setRange(1, 10000)
        self.points.setValue(100)
        
        self.w_edge = QSpinBox()
        self.w_edge.setRange(1, 10)
        self.w_edge.setValue(100)
        
        self.convergence = QSpinBox()
        self.convergence.setRange(0, 1)
        self.convergence.setValue(0)
        
        self.btn_run_snake = QPushButton("Run Active Contour")
        self.btn_run_snake.clicked.connect(parent.run_active_contour)

        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center X"))
        center_layout.addWidget(self.centerX)
        center_layout.addWidget(QLabel("Center Y"))
        center_layout.addWidget(self.centerY)
        active_contour_layout.addLayout(center_layout)

        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius"))
        radius_layout.addWidget(self.radius)
        active_contour_layout.addLayout(radius_layout)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha"))
        alpha_layout.addWidget(self.alpha)
        active_contour_layout.addLayout(alpha_layout)

        beta_layout = QHBoxLayout()
        beta_layout.addWidget(QLabel("Beta"))
        beta_layout.addWidget(self.beta)
        active_contour_layout.addLayout(beta_layout)

        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma"))
        gamma_layout.addWidget(self.gamma)
        active_contour_layout.addLayout(gamma_layout)

        iterations_layout = QHBoxLayout()
        iterations_layout.addWidget(QLabel("Iterations"))
        iterations_layout.addWidget(self.iterations)
        active_contour_layout.addLayout(iterations_layout)

        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("Points"))
        points_layout.addWidget(self.points)
        active_contour_layout.addLayout(points_layout)
        
        w_edge_layout = QHBoxLayout()
        w_edge_layout.addWidget(QLabel("Edge weight"))
        w_edge_layout.addWidget(self.w_edge)
        active_contour_layout.addLayout(w_edge_layout)

        convergence_layout = QHBoxLayout()
        convergence_layout.addWidget(QLabel("Convergance"))
        convergence_layout.addWidget(self.convergence)
        active_contour_layout.addLayout(convergence_layout)

        active_contour_layout.addWidget(self.btn_run_snake)

        main_layout.addWidget(active_contour_frame)

class HoughTransformTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create a scroll area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide horizontal scrollbar
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # Hide vertical scrollbar

        # Create a widget to hold the content
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        # Main layout for the tab
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

        # Layout for the content inside the scroll area
        hough_transform_layout = QVBoxLayout(content_widget)

        ############################## Shared Canny Detector Parameters ##############################
        canny_group_frame = QFrame()
        canny_group_frame.setObjectName("canny_group_frame")
        canny_layout = QVBoxLayout(canny_group_frame)

        self.cannyLowThreshold = QSpinBox()
        self.cannyLowThreshold.setRange(0, 255)
        self.cannyLowThreshold.setValue(50)

        self.cannyHighThreshold = QSpinBox()
        self.cannyHighThreshold.setRange(0, 255)
        self.cannyHighThreshold.setValue(150)

        self.cannyBlurKSize = QSpinBox()
        self.cannyBlurKSize.setRange(1, 15)
        self.cannyBlurKSize.setValue(5)

        low_threshold_layout = QHBoxLayout()
        low_threshold_layout.addWidget(QLabel("Canny Low Threshold"))
        low_threshold_layout.addWidget(self.cannyLowThreshold)
        canny_layout.addLayout(low_threshold_layout)

        high_threshold_layout = QHBoxLayout()
        high_threshold_layout.addWidget(QLabel("Canny High Threshold"))
        high_threshold_layout.addWidget(self.cannyHighThreshold)
        canny_layout.addLayout(high_threshold_layout)

        blur_ksize_layout = QHBoxLayout()
        blur_ksize_layout.addWidget(QLabel("Blur Kernel Size"))
        blur_ksize_layout.addWidget(self.cannyBlurKSize)
        canny_layout.addLayout(blur_ksize_layout)

        hough_transform_layout.addWidget(canny_group_frame)

        ############################## Line Detection Parameters ##############################
        line_group_frame = QFrame()
        line_group_frame.setObjectName("line_group_frame")
        line_layout = QVBoxLayout(line_group_frame)

        self.numRho = QSpinBox()
        self.numRho.setRange(1, 500)
        self.numRho.setValue(180)

        self.numTheta = QSpinBox()
        self.numTheta.setRange(1, 500)
        self.numTheta.setValue(180)

        self.houghThresholdRatio = QDoubleSpinBox()
        self.houghThresholdRatio.setRange(0.0, 1.0)
        self.houghThresholdRatio.setSingleStep(0.1)
        self.houghThresholdRatio.setValue(0.6)

        self.btn_detect_lines = QPushButton("Detect Lines")
        self.btn_detect_lines.clicked.connect(parent.detect_lines)

        num_rho_layout = QHBoxLayout()
        num_rho_layout.addWidget(QLabel("Num Rho"))
        num_rho_layout.addWidget(self.numRho)
        line_layout.addLayout(num_rho_layout)

        num_theta_layout = QHBoxLayout()
        num_theta_layout.addWidget(QLabel("Num Theta"))
        num_theta_layout.addWidget(self.numTheta)
        line_layout.addLayout(num_theta_layout)

        hough_threshold_ratio_layout = QHBoxLayout()
        hough_threshold_ratio_layout.addWidget(QLabel("Hough Threshold Ratio"))
        hough_threshold_ratio_layout.addWidget(self.houghThresholdRatio)
        line_layout.addLayout(hough_threshold_ratio_layout)

        line_layout.addWidget(self.btn_detect_lines)

        hough_transform_layout.addWidget(line_group_frame)

        ############################## Circle Detection Parameters ##############################
        circle_group_frame = QFrame()
        circle_group_frame.setObjectName("circle_group_frame")
        circle_layout = QVBoxLayout(circle_group_frame)

        self.rMin = QSpinBox()
        self.rMin.setRange(1, 100)
        self.rMin.setValue(20)

        self.rMax = QSpinBox()
        self.rMax.setRange(1, 500)
        self.rMax.setValue(100)

        self.numThetas = QSpinBox()
        self.numThetas.setRange(1, 360)
        self.numThetas.setValue(50)

        self.btn_detect_circles = QPushButton("Detect Circles")
        self.btn_detect_circles.clicked.connect(parent.detect_circles)

        r_min_layout = QHBoxLayout()
        r_min_layout.addWidget(QLabel("Min Radius"))
        r_min_layout.addWidget(self.rMin)
        circle_layout.addLayout(r_min_layout)

        r_max_layout = QHBoxLayout()
        r_max_layout.addWidget(QLabel("Max Radius"))
        r_max_layout.addWidget(self.rMax)
        circle_layout.addLayout(r_max_layout)

        num_thetas_layout = QHBoxLayout()
        num_thetas_layout.addWidget(QLabel("Num Thetas"))
        num_thetas_layout.addWidget(self.numThetas)
        circle_layout.addLayout(num_thetas_layout)

        circle_layout.addWidget(self.btn_detect_circles)

        hough_transform_layout.addWidget(circle_group_frame)

        ############################## Ellipse Detection Parameters ##############################
        ellipse_group_frame = QFrame()
        ellipse_group_frame.setObjectName("ellipse_group_frame")
        ellipse_layout = QVBoxLayout(ellipse_group_frame)

        self.aMin = QSpinBox()
        self.aMin.setRange(1, 500)
        self.aMin.setValue(20)

        self.aMax = QSpinBox()
        self.aMax.setRange(1, 500)
        self.aMax.setValue(100)

        self.bMin = QSpinBox()
        self.bMin.setRange(1, 500)
        self.bMin.setValue(10)

        self.bMax = QSpinBox()
        self.bMax.setRange(1, 500)
        self.bMax.setValue(50)

        self.thetaStep = QSpinBox()
        self.thetaStep.setRange(1, 180)
        self.thetaStep.setValue(10)

        self.ellipseThresholdRatio = QDoubleSpinBox()
        self.ellipseThresholdRatio.setRange(0.0, 1.0)
        self.ellipseThresholdRatio.setSingleStep(0.1)
        self.ellipseThresholdRatio.setValue(0.5)

        self.minDist = QSpinBox()
        self.minDist.setRange(1, 100)
        self.minDist.setValue(20)

        self.btn_detect_ellipses = QPushButton("Detect Ellipses")
        # self.btn_detect_ellipses.clicked.connect(parent.detect_ellipses)

        a_min_layout = QHBoxLayout()
        a_min_layout.addWidget(QLabel("Min Semi-Major Axis (aMin)"))
        a_min_layout.addWidget(self.aMin)
        ellipse_layout.addLayout(a_min_layout)

        a_max_layout = QHBoxLayout()
        a_max_layout.addWidget(QLabel("Max Semi-Major Axis (aMax)"))
        a_max_layout.addWidget(self.aMax)
        ellipse_layout.addLayout(a_max_layout)

        b_min_layout = QHBoxLayout()
        b_min_layout.addWidget(QLabel("Min Semi-Minor Axis (bMin)"))
        b_min_layout.addWidget(self.bMin)
        ellipse_layout.addLayout(b_min_layout)

        b_max_layout = QHBoxLayout()
        b_max_layout.addWidget(QLabel("Max Semi-Minor Axis (bMax)"))
        b_max_layout.addWidget(self.bMax)
        ellipse_layout.addLayout(b_max_layout)

        theta_step_layout = QHBoxLayout()
        theta_step_layout.addWidget(QLabel("Theta Step"))
        theta_step_layout.addWidget(self.thetaStep)
        ellipse_layout.addLayout(theta_step_layout)

        ellipse_threshold_ratio_layout = QHBoxLayout()
        ellipse_threshold_ratio_layout.addWidget(QLabel("Threshold Ratio"))
        ellipse_threshold_ratio_layout.addWidget(self.ellipseThresholdRatio)
        ellipse_layout.addLayout(ellipse_threshold_ratio_layout)

        min_dist_layout = QHBoxLayout()
        min_dist_layout.addWidget(QLabel("Min Distance Between Ellipses"))
        min_dist_layout.addWidget(self.minDist)
        ellipse_layout.addLayout(min_dist_layout)

        ellipse_layout.addWidget(self.btn_detect_ellipses)

        hough_transform_layout.addWidget(ellipse_group_frame)

class NoiseFilterTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        noise_and_filter_layout = QVBoxLayout(self)

        # Noise Frame
        noise_frame = QFrame()
        noise_frame.setObjectName("noise_frame")
        noise_layout = QVBoxLayout(noise_frame)
        noise_layout.setAlignment(Qt.AlignTop)

        # Noise UI Components
        self.noiseType = QComboBox()
        self.noiseType.addItems(["uniform", "gaussian", "salt_pepper"])
        self.noiseType.currentTextChanged.connect(self.update_noise_params_visibility)

        self.noiseIntensity = QSpinBox()
        self.noiseIntensity.setRange(1, 100)
        self.noiseIntensity.setValue(50)

        self.gaussianMean = QDoubleSpinBox()
        self.gaussianMean.setRange(-50, 50)
        self.gaussianMean.setValue(0)

        self.gaussianStd = QDoubleSpinBox()
        self.gaussianStd.setRange(1, 100)
        self.gaussianStd.setValue(25)

        self.saltProb = QDoubleSpinBox()
        self.saltProb.setRange(0.0, 1.0)
        self.saltProb.setSingleStep(0.01)
        self.saltProb.setValue(0.02)

        self.pepperProb = QDoubleSpinBox()
        self.pepperProb.setRange(0.0, 1.0)
        self.pepperProb.setSingleStep(0.01)
        self.pepperProb.setValue(0.02)

        self.btn_noise = QPushButton("Add Noise")
        self.btn_noise.clicked.connect(parent.apply_noise)

        self.noiseParamLayout = QVBoxLayout()
        noise_type_layout = QHBoxLayout()
        noise_type_layout.addWidget(QLabel("Noise Type"))
        noise_type_layout.addWidget(self.noiseType)
        self.noiseParamLayout.addLayout(noise_type_layout)

        self.intensity_label = QLabel("Intensity")
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(self.intensity_label)
        intensity_layout.addWidget(self.noiseIntensity)
        self.noiseParamLayout.addLayout(intensity_layout)

        self.mean_label = QLabel("Mean")
        mean_layout = QHBoxLayout()
        mean_layout.addWidget(self.mean_label)
        mean_layout.addWidget(self.gaussianMean)
        self.noiseParamLayout.addLayout(mean_layout)

        self.std_label = QLabel("Std Dev")
        std_layout = QHBoxLayout()
        std_layout.addWidget(self.std_label)
        std_layout.addWidget(self.gaussianStd)
        self.noiseParamLayout.addLayout(std_layout)

        self.salt_label = QLabel("Salt Prob")
        salt_layout = QHBoxLayout()
        salt_layout.addWidget(self.salt_label)
        salt_layout.addWidget(self.saltProb)
        self.noiseParamLayout.addLayout(salt_layout)

        self.pepper_label = QLabel("Pepper Prob")
        pepper_layout = QHBoxLayout()
        pepper_layout.addWidget(self.pepper_label)
        pepper_layout.addWidget(self.pepperProb)
        self.noiseParamLayout.addLayout(pepper_layout)

        self.noiseParamLayout.addWidget(self.btn_noise)

        noise_layout.addLayout(self.noiseParamLayout)
        noise_and_filter_layout.addWidget(noise_frame)

        # Filter Frame
        filter_frame = QFrame()
        filter_frame.setObjectName("filter_frame")
        filter_layout = QVBoxLayout(filter_frame)
        filter_layout.setAlignment(Qt.AlignTop)

        # Filter UI Components
        self.filterType = QComboBox()
        self.filterType.addItems(["average", "gaussian", "median"])

        self.kernelSize = QSpinBox()
        self.kernelSize.setRange(1, 15)
        self.kernelSize.setValue(3)

        self.sigmaValue = QDoubleSpinBox()
        self.sigmaValue.setRange(0.1, 10.0)
        self.sigmaValue.setValue(1.0)

        self.btn_filter = QPushButton("Apply Filter")
        self.btn_filter.clicked.connect(parent.apply_filter)

        self.filterParamLayout = QVBoxLayout()
        filter_type_layout = QHBoxLayout()
        filter_type_layout.addWidget(QLabel("Filter Type"))
        filter_type_layout.addWidget(self.filterType)
        self.filterParamLayout.addLayout(filter_type_layout)

        kernel_size_layout = QHBoxLayout()
        kernel_size_layout.addWidget(QLabel("Kernel Size"))
        kernel_size_layout.addWidget(self.kernelSize)
        self.filterParamLayout.addLayout(kernel_size_layout)

        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma"))
        sigma_layout.addWidget(self.sigmaValue)
        self.filterParamLayout.addLayout(sigma_layout)

        self.filterParamLayout.addWidget(self.btn_filter)

        filter_layout.addLayout(self.filterParamLayout)
        noise_and_filter_layout.addWidget(filter_frame)

        self.update_noise_params_visibility()

    def update_noise_params_visibility(self):
        noise_type = self.noiseType.currentText()
        self.noiseIntensity.setVisible(noise_type == "uniform")
        self.intensity_label.setVisible(noise_type == "uniform")
        self.gaussianMean.setVisible(noise_type == "gaussian")
        self.mean_label.setVisible(noise_type == "gaussian")
        self.gaussianStd.setVisible(noise_type == "gaussian")
        self.std_label.setVisible(noise_type == "gaussian")
        self.saltProb.setVisible(noise_type == "salt_pepper")
        self.salt_label.setVisible(noise_type == "salt_pepper")
        self.pepperProb.setVisible(noise_type == "salt_pepper")
        self.pepper_label.setVisible(noise_type == "salt_pepper")

class EdgeDetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        edge_detection_frame = QFrame()
        edge_detection_frame.setObjectName("edge_detection_frame")
        edge_detection_layout = QVBoxLayout(edge_detection_frame)
        edge_detection_layout.setAlignment(Qt.AlignTop)

        self.edgeType = QComboBox()
        self.edgeType.addItems(["sobel", "canny", "prewitt", "roberts"])
        self.edgeType.currentTextChanged.connect(self.update_edge_params_visibility)

        self.sobelKernelSize = QSpinBox()
        self.sobelKernelSize.setRange(1, 15)
        self.sobelKernelSize.setValue(3)

        self.sobelSigma = QDoubleSpinBox()
        self.sobelSigma.setRange(0.1, 10.0)
        self.sobelSigma.setValue(1.0)

        self.cannyLowThreshold = QSpinBox()
        self.cannyLowThreshold.setRange(0, 255)
        self.cannyLowThreshold.setValue(50)

        self.cannyHighThreshold = QSpinBox()
        self.cannyHighThreshold.setRange(0, 255)
        self.cannyHighThreshold.setValue(150)

        self.cannyMaxEdgeVal = QSpinBox()
        self.cannyMaxEdgeVal.setRange(0, 255)
        self.cannyMaxEdgeVal.setValue(255)

        self.cannyMinEdgeVal = QSpinBox()
        self.cannyMinEdgeVal.setRange(0, 255)
        self.cannyMinEdgeVal.setValue(0)

        self.prewittThreshold = QSpinBox()
        self.prewittThreshold.setRange(0, 255)
        self.prewittThreshold.setValue(50)

        self.prewittValue = QSpinBox()
        self.prewittValue.setRange(0, 255)
        self.prewittValue.setValue(255)

        self.btn_edge_detection = QPushButton("Detect Edges")
        self.btn_edge_detection.clicked.connect(parent.detect_edges)

        self.edgeParamLayout = QVBoxLayout()
        edge_type_layout = QHBoxLayout()
        edge_type_layout.addWidget(QLabel("Edge Type"))
        edge_type_layout.addWidget(self.edgeType)
        self.edgeParamLayout.addLayout(edge_type_layout)

        self.sobel_kernel_label = QLabel("Kernel Size")
        sobel_kernel_layout = QHBoxLayout()
        sobel_kernel_layout.addWidget(self.sobel_kernel_label)
        sobel_kernel_layout.addWidget(self.sobelKernelSize)
        self.edgeParamLayout.addLayout(sobel_kernel_layout)

        self.sobel_sigma_label = QLabel("Sigma")
        sobel_sigma_layout = QHBoxLayout()
        sobel_sigma_layout.addWidget(self.sobel_sigma_label)
        sobel_sigma_layout.addWidget(self.sobelSigma)
        self.edgeParamLayout.addLayout(sobel_sigma_layout)

        self.canny_low_label = QLabel("Low Threshold")
        canny_low_layout = QHBoxLayout()
        canny_low_layout.addWidget(self.canny_low_label)
        canny_low_layout.addWidget(self.cannyLowThreshold)
        self.edgeParamLayout.addLayout(canny_low_layout)

        self.canny_high_label = QLabel("High Threshold")
        canny_high_layout = QHBoxLayout()
        canny_high_layout.addWidget(self.canny_high_label)
        canny_high_layout.addWidget(self.cannyHighThreshold)
        self.edgeParamLayout.addLayout(canny_high_layout)

        self.canny_max_label = QLabel("Max Edge")
        canny_max_layout = QHBoxLayout()
        canny_max_layout.addWidget(self.canny_max_label)
        canny_max_layout.addWidget(self.cannyMaxEdgeVal)
        self.edgeParamLayout.addLayout(canny_max_layout)

        self.canny_min_label = QLabel("Min Edge")
        canny_min_layout = QHBoxLayout()
        canny_min_layout.addWidget(self.canny_min_label)
        canny_min_layout.addWidget(self.cannyMinEdgeVal)
        self.edgeParamLayout.addLayout(canny_min_layout)

        self.prewitt_threshold_label = QLabel("Threshold")
        prewitt_threshold_layout = QHBoxLayout()
        prewitt_threshold_layout.addWidget(self.prewitt_threshold_label)
        prewitt_threshold_layout.addWidget(self.prewittThreshold)
        self.edgeParamLayout.addLayout(prewitt_threshold_layout)

        self.prewitt_value_label = QLabel("Value")
        prewitt_value_layout = QHBoxLayout()
        prewitt_value_layout.addWidget(self.prewitt_value_label)
        prewitt_value_layout.addWidget(self.prewittValue)
        self.edgeParamLayout.addLayout(prewitt_value_layout)

        self.edgeParamLayout.addWidget(self.btn_edge_detection)

        edge_detection_layout.addLayout(self.edgeParamLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(edge_detection_frame)

        self.update_edge_params_visibility()

    def update_edge_params_visibility(self):
        edge_type = self.edgeType.currentText()
        is_sobel = edge_type == "sobel"
        is_canny = edge_type == "canny"
        is_prewitt = edge_type == "prewitt"
        is_roberts = edge_type == "roberts"

        self.sobelKernelSize.setVisible(is_sobel)
        self.sobel_kernel_label.setVisible(is_sobel)
        self.sobelSigma.setVisible(is_sobel)
        self.sobel_sigma_label.setVisible(is_sobel)

        self.cannyLowThreshold.setVisible(is_canny)
        self.canny_low_label.setVisible(is_canny)
        self.cannyHighThreshold.setVisible(is_canny)
        self.canny_high_label.setVisible(is_canny)
        self.cannyMaxEdgeVal.setVisible(is_canny)
        self.canny_max_label.setVisible(is_canny)
        self.cannyMinEdgeVal.setVisible(is_canny)
        self.canny_min_label.setVisible(is_canny)

        self.prewittThreshold.setVisible(is_prewitt)
        self.prewitt_threshold_label.setVisible(is_prewitt)
        self.prewittValue.setVisible(is_prewitt)
        self.prewitt_value_label.setVisible(is_prewitt)

class ThresholdingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        thresholding_frame = QFrame()
        thresholding_frame.setObjectName("thresholding_frame")
        thresholding_layout = QVBoxLayout(thresholding_frame)
        thresholding_layout.setAlignment(Qt.AlignTop)

        self.thresholdType = QComboBox()
        self.thresholdType.addItems(["global", "local"])
        self.thresholdType.currentTextChanged.connect(self.update_threshold_params_visibility)

        self.globalThreshold = QSpinBox()
        self.globalThreshold.setRange(0, 255)
        self.globalThreshold.setValue(128)

        self.kernelSizeThreshold = QSpinBox()
        self.kernelSizeThreshold.setRange(1, 15)
        self.kernelSizeThreshold.setValue(4)

        self.kValue = QDoubleSpinBox()
        self.kValue.setRange(0.0, 5.0)
        self.kValue.setSingleStep(0.1)
        self.kValue.setValue(2.0)

        self.btn_threshold = QPushButton("Apply Thresholding")
        self.btn_threshold.clicked.connect(parent.apply_thresholding)

        self.thresholdLayout = QVBoxLayout()
        threshold_type_layout = QHBoxLayout()
        threshold_type_layout.addWidget(QLabel("Threshold Type"))
        threshold_type_layout.addWidget(self.thresholdType)
        self.thresholdLayout.addLayout(threshold_type_layout)

        self.global_threshold_label = QLabel("Global Threshold")
        global_threshold_layout = QHBoxLayout()
        global_threshold_layout.addWidget(self.global_threshold_label)
        global_threshold_layout.addWidget(self.globalThreshold)
        self.thresholdLayout.addLayout(global_threshold_layout)

        self.kernel_size_label = QLabel("Kernel Size")
        kernel_size_threshold_layout = QHBoxLayout()
        kernel_size_threshold_layout.addWidget(self.kernel_size_label)
        kernel_size_threshold_layout.addWidget(self.kernelSizeThreshold)
        self.thresholdLayout.addLayout(kernel_size_threshold_layout)

        self.k_value_label = QLabel("K Value")
        k_value_layout = QHBoxLayout()
        k_value_layout.addWidget(self.k_value_label)
        k_value_layout.addWidget(self.kValue)
        self.thresholdLayout.addLayout(k_value_layout)

        self.thresholdLayout.addWidget(self.btn_threshold)

        thresholding_layout.addLayout(self.thresholdLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(thresholding_frame)

        self.update_threshold_params_visibility()

    def update_threshold_params_visibility(self):
        threshold_type = self.thresholdType.currentText()
        is_global = threshold_type == "global"
        is_local = threshold_type == "local"

        self.globalThreshold.setVisible(is_global)
        self.global_threshold_label.setVisible(is_global)
        self.kernelSizeThreshold.setVisible(is_local)
        self.kernel_size_label.setVisible(is_local)
        self.kValue.setVisible(is_local)
        self.k_value_label.setVisible(is_local)

class FrequencyFilterTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        frequency_filter_frame = QFrame()
        frequency_filter_frame.setObjectName("frequency_filter_frame")
        frequency_filter_layout = QVBoxLayout(frequency_filter_frame)
        frequency_filter_layout.setAlignment(Qt.AlignTop)

        self.freqType = QComboBox()
        self.freqType.addItems(["low_pass", "high_pass"])

        self.freqRadius = QSpinBox()
        self.freqRadius.setRange(1, 100)
        self.freqRadius.setValue(10)

        self.btn_freq_filter = QPushButton("Apply Frequency Filter")
        self.btn_freq_filter.clicked.connect(parent.apply_frequency_filter)

        self.freqLayout = QVBoxLayout()
        freq_type_layout = QHBoxLayout()
        freq_type_layout.addWidget(QLabel("Filter Type"))
        freq_type_layout.addWidget(self.freqType)
        self.freqLayout.addLayout(freq_type_layout)

        freq_radius_layout = QHBoxLayout()
        freq_radius_layout.addWidget(QLabel("Radius"))
        freq_radius_layout.addWidget(self.freqRadius)
        self.freqLayout.addLayout(freq_radius_layout)

        self.freqLayout.addWidget(self.btn_freq_filter)

        frequency_filter_layout.addLayout(self.freqLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frequency_filter_frame)

class HybridImageTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.parent = parent
        self.image1 = None
        self.image2 = None

        hybrid_image_frame = QFrame()
        hybrid_image_frame.setObjectName("hybrid_image_frame")
        hybrid_image_layout = QVBoxLayout(hybrid_image_frame)
        hybrid_image_layout.setAlignment(Qt.AlignTop)

        self.image1_label = QLabel("No Image 1 Loaded")
        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image1_label.setStyleSheet("border: 1px solid rgb(40, 57, 153);")
        self.image1_label.setFixedSize(330, 330)
        self.image1_label.mouseDoubleClickEvent = lambda event: self.load_image(1)

        self.image2_label = QLabel("No Image 2 Loaded")
        self.image2_label.setAlignment(Qt.AlignCenter)
        self.image2_label.setStyleSheet("border: 1px solid rgb(40, 57, 153);")
        self.image2_label.setFixedSize(330, 330)
        self.image2_label.mouseDoubleClickEvent = lambda event: self.load_image(2)

        self.cutoff1 = QSpinBox()
        self.cutoff1.setRange(1, 100)
        self.cutoff1.setValue(10)

        self.cutoff2 = QSpinBox()
        self.cutoff2.setRange(1, 100)
        self.cutoff2.setValue(10)

        self.type1 = QComboBox()
        self.type1.addItems(["lp", "hp"])

        self.type2 = QComboBox()
        self.type2.addItems(["lp", "hp"])

        self.btn_hybrid = QPushButton("Create Hybrid Image")
        self.btn_hybrid.clicked.connect(self.create_hybrid_image)

        self.hybridLayout = QVBoxLayout()

        cutoff1_layout = QHBoxLayout()
        cutoff1_layout.addWidget(QLabel("Cutoff 1"))
        cutoff1_layout.addWidget(self.cutoff1)
        self.hybridLayout.addLayout(cutoff1_layout)

        type1_layout = QHBoxLayout()
        type1_layout.addWidget(QLabel("Type 1"))
        type1_layout.addWidget(self.type1)
        self.hybridLayout.addLayout(type1_layout)

        cutoff2_layout = QHBoxLayout()
        cutoff2_layout.addWidget(QLabel("Cutoff 2"))
        cutoff2_layout.addWidget(self.cutoff2)
        self.hybridLayout.addLayout(cutoff2_layout)

        type2_layout = QHBoxLayout()
        type2_layout.addWidget(QLabel("Type 2"))
        type2_layout.addWidget(self.type2)
        self.hybridLayout.addLayout(type2_layout)

        self.hybridLayout.addWidget(self.btn_hybrid)
        
        images_layout = QVBoxLayout()
        images_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        images_layout.addSpacing(10)
        images_layout.addWidget(self.image1_label)
        images_layout.addWidget(self.image2_label)
        self.hybridLayout.addLayout(images_layout)

        hybrid_image_layout.addLayout(self.hybridLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(hybrid_image_frame)

    def load_image(self, image_number):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            image = cv2.imread(file_path)
            if image_number == 1:
                self.image1 = image
                self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
                self.display_image(self.image1, self.image1_label)
            elif image_number == 2:
                
                self.image2 = image
                self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
                self.display_image(self.image2, self.image2_label)

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def create_hybrid_image(self):
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Warning", "Please load both images first.")
            return

        cutoff1 = self.cutoff1.value()
        cutoff2 = self.cutoff2.value()
        type1 = self.type1.currentText()
        type2 = self.type2.currentText()

        hybrid_image = self.parent.processors['frequency'].create_hybrid_image(self.image1, self.image2, cutoff1, cutoff2, type1, type2)
        self.parent.display_image(hybrid_image)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Image Processing App")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout()
        central_widget.setLayout(self.main_layout)
        
        self.init_ui(self.main_layout)

        # Single data structure to store all parameters
        self.params = {
            "noise_filter": {},
            "filtering": {},
            "edge_detection": {},
            "thresholding": {},
            "frequency_filter": {},
            "hybrid_image": {},
            "shape_detection":{},
            "active_contour":{}
            
        }
        
        self.connect_signals()
        # Image & Processor Variables
        self.image = None
        self.original_image = None
        self.modified_image = None
        self.processors = {key: ProcessorFactory.create_processor(key) for key in ['noise', 'edge_detector', 'thresholding', 'frequency', 'histogram', 'image', 'active_contour']}

    def run_active_contour(self):
        """
        Runs the Active Contour (Snake) algorithm on the loaded image.
        """
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        contour_params = self.params["active_contour"]

        # Initialize the snake
        snake, history = self.processors["active_contour"].detect_contour(**contour_params)

        plt.imshow(self.image, cmap='gray')
        plt.plot(snake[:, 0], snake[:, 1], 'r-', label="Snake Contour")
        plt.title("Active Contour Result")
        plt.legend()
        plt.show()
        # # Ensure the viewer exists before setting history
        # if not hasattr(self, 'active_contour_viewer'):
        #     self.active_contour_viewer = ActiveContourVisualizationWidget(self,self.processors["active_contour"])
        #     self.image_display_layout.addWidget(self.active_contour_viewer)  # Add widget to layout

        # # Update visualization with the contour history
        # self.active_contour_viewer.set_history(history)
        # self.active_contour_viewer.show()  # Ensure the widget is displayed 
            


    def init_ui(self, main_layout):
        left_frame = QFrame()
        left_frame.setObjectName("left_frame")
        left_layout = QVBoxLayout(left_frame)
        
        tab_widget = QTabWidget()
        tab_widget.setObjectName("tab_widget")

        # Noise and Filter Tab
        self.noise_filter_tab = NoiseFilterTab(self)
        tab_widget.addTab(self.noise_filter_tab, "Noise & Filter")

        # Edge Detection Tab
        self.edge_detection_tab = EdgeDetectionTab(self)
        tab_widget.addTab(self.edge_detection_tab, "Edge Detection")

        # Thresholding Tab
        self.thresholding_tab = ThresholdingTab(self)
        tab_widget.addTab(self.thresholding_tab, "Thresholding")

        # Frequency Filter Tab
        self.frequency_filter_tab = FrequencyFilterTab(self)
        tab_widget.addTab(self.frequency_filter_tab, "Frequency Filter")

        # Hybrid Image Tab
        self.hybrid_image_tab = HybridImageTab(self)
        tab_widget.addTab(self.hybrid_image_tab, "Hybrid Image")

        # Hough Transform Tab
        self.hough_transform_tab = HoughTransformTab(self)
        tab_widget.addTab(self.hough_transform_tab, "Hough Transform")

        self.active_contour_tab = ActiveContourTab(self)
        tab_widget.addTab(self.active_contour_tab, "Active Contour")

        
        left_layout.addWidget(tab_widget)
        main_layout.addWidget(left_frame)
        
        #? Right Frame with Control Buttons and Image Display
        right_frame = QFrame()
        right_frame.setObjectName("right_frame")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        
        # Control Buttons Frame
        control_frame = QFrame()
        control_frame.setMaximumHeight(100)
        control_layout = QHBoxLayout(control_frame)

        # Control Buttons Frame
        control_buttons_frame = QFrame()
        control_buttons_layout = QHBoxLayout(control_buttons_frame)

        self.btn_histogram = QPushButton()
        self.btn_histogram.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/diagram-bar-3.png')))
        self.btn_histogram.setIconSize(QSize(32, 32))
        self.btn_histogram.clicked.connect(self.show_histogram)
        control_buttons_layout.addWidget(self.btn_histogram)

        self.btn_equalize = QPushButton()
        self.btn_equalize.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/equalizer-solid.png')))
        self.btn_equalize.setIconSize(QSize(32, 32))
        self.btn_equalize.clicked.connect(self.equalize)
        control_buttons_layout.addWidget(self.btn_equalize)

        self.btn_normalize = QPushButton()
        self.btn_normalize.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/gaussain-curve.png')))
        self.btn_normalize.setIconSize(QSize(32, 32))
        self.btn_normalize.clicked.connect(self.normalize)
        control_buttons_layout.addWidget(self.btn_normalize)

        control_layout.addWidget(control_buttons_frame)

        control_layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # Image Control Buttons Frame
        image_control_buttons_frame = QFrame()
        image_control_buttons_layout = QHBoxLayout(image_control_buttons_frame)

        self.btn_confirm = QPushButton()
        self.btn_confirm.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/confirm.png')))
        self.btn_confirm.setIconSize(QSize(28, 28))
        self.btn_confirm.clicked.connect(self.confirm_edit)
        image_control_buttons_layout.addWidget(self.btn_confirm)

        self.btn_discard = QPushButton()
        self.btn_discard.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/discard.png')))
        self.btn_discard.setIconSize(QSize(28, 28))
        self.btn_discard.clicked.connect(self.discard_edit)
        image_control_buttons_layout.addWidget(self.btn_discard)

        self.btn_reset = QPushButton()
        self.btn_reset.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/reset.png')))
        self.btn_reset.setIconSize(QSize(28, 28))
        self.btn_reset.clicked.connect(self.reset_image)
        image_control_buttons_layout.addWidget(self.btn_reset)
        control_layout.addWidget(image_control_buttons_frame)

        # Add the control frame to the right layout
        right_layout.addWidget(control_frame)

        # Image Display Frame
        image_display_frame = QFrame()

        image_display_frame.setFixedSize(1390,880)
        self.image_display_layout = QVBoxLayout(image_display_frame)

        self.lbl_image = QLabel("No Image Loaded")
        self.lbl_image.setObjectName("lbl_image")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.image_display_layout.addWidget(self.lbl_image)
        self.lbl_image.mouseDoubleClickEvent = self.on_image_label_double_click


        # Add the image display frame to the right layout
        right_layout.addWidget(image_display_frame)

        # Add the right frame to the main layout
        main_layout.addWidget(right_frame)
    
    def connect_signals(self):
        # Active contour tab
        
        active_contour_ui = {
            "center" : (self.active_contour_tab.centerX, self.active_contour_tab.centerY),
            "radius" : self.active_contour_tab.radius,
            "alpha" : self.active_contour_tab.alpha,
            "beta" : self.active_contour_tab.beta,
            "gamma": self.active_contour_tab.gamma,
            "iterations" : self.active_contour_tab.iterations,
            "points": self.active_contour_tab.points,
            "w_edge" : self.active_contour_tab.w_edge,
            "convergence" : self.active_contour_tab.convergence
            
         } 
        
        for widget in active_contour_ui.values():
            if isinstance(widget, QComboBox):
                # Connect QComboBox's currentTextChanged signal
                widget.currentTextChanged.connect(lambda: self.update_params("active_contour", active_contour_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Connect QSpinBox/QDoubleSpinBox's valueChanged signal
                widget.valueChanged.connect(lambda: self.update_params("active_contour", active_contour_ui))      

        # Hough Transform and Shape Detection Tab
        shape_detection_ui = {
            'num_rho': self.hough_transform_tab.numRho,
            'num_theta': self.hough_transform_tab.numTheta,
            'hough_threshold_ratio': self.hough_transform_tab.houghThresholdRatio,
            'r_min': self.hough_transform_tab.rMin,
            'r_max': self.hough_transform_tab.rMax,
            'num_thetas': self.hough_transform_tab.numThetas,
            # Shared Canny Detector Parameters
            'canny_low_threshold': self.hough_transform_tab.cannyLowThreshold,
            'canny_high_threshold': self.hough_transform_tab.cannyHighThreshold,
            'canny_blur_ksize': self.hough_transform_tab.cannyBlurKSize
        }

        for widget in shape_detection_ui.values():
            if isinstance(widget, QComboBox):
                # Connect QComboBox's currentTextChanged signal
                widget.currentTextChanged.connect(lambda: self.update_params("shape_detection", shape_detection_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Connect QSpinBox/QDoubleSpinBox's valueChanged signal
                widget.valueChanged.connect(lambda: self.update_params("shape_detection", shape_detection_ui))      

        
        # Noise & Filter Tab
        noise_filter_ui = {
            "noise_type": self.noise_filter_tab.noiseType,
            "noise_intensity": self.noise_filter_tab.noiseIntensity,
            "gaussian_mean": self.noise_filter_tab.gaussianMean,
            "gaussian_std": self.noise_filter_tab.gaussianStd,
            "salt_prob": self.noise_filter_tab.saltProb,
            "pepper_prob": self.noise_filter_tab.pepperProb
        }
        
        for widget in noise_filter_ui.values():
            if isinstance(widget, QComboBox):
                # Connect QComboBox's currentTextChanged signal
                widget.currentTextChanged.connect(lambda: self.update_params("noise_filter", noise_filter_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Connect QSpinBox/QDoubleSpinBox's valueChanged signal
                widget.valueChanged.connect(lambda: self.update_params("noise_filter", noise_filter_ui))      
        
        
        # Noise & Filter Tab - Filter UI Components
        filter_ui = {
            "filter_type": self.noise_filter_tab.filterType,
            "kernel_size": self.noise_filter_tab.kernelSize,
            "sigma_value": self.noise_filter_tab.sigmaValue
        }
        for widget in filter_ui.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.update_params("filter", filter_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.update_params("filter", filter_ui))

        # Edge Detection Tab
        self.edge_detection_ui = {
            "detector_method": self.edge_detection_tab.edgeType,
            "kernel_size": self.edge_detection_tab.sobelKernelSize,
            "sigma": self.edge_detection_tab.sobelSigma,
            "low_threshold": self.edge_detection_tab.cannyLowThreshold,
            "high_threshold": self.edge_detection_tab.cannyHighThreshold,
            "max_edge_val": self.edge_detection_tab.cannyMaxEdgeVal,
            "min_edge_val": self.edge_detection_tab.cannyMinEdgeVal,
            "threshold": self.edge_detection_tab.prewittThreshold,
            "value": self.edge_detection_tab.prewittValue
        }
        # Edge Detection Tab
        for widget in self.edge_detection_ui.values():
            if isinstance(widget, QComboBox):
                # Connect QComboBox's currentTextChanged signal
                widget.currentTextChanged.connect(lambda: self.update_params("edge_detection", self.edge_detection_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Connect QSpinBox/QDoubleSpinBox's valueChanged signal
                widget.valueChanged.connect(lambda: self.update_params("edge_detection", self.edge_detection_ui))
 
        # Thresholding Tab
        self.thresholding_ui = {
            "threshold_type": self.thresholding_tab.thresholdType,
            "T": self.thresholding_tab.globalThreshold,
            "kernal": self.thresholding_tab.kernelSizeThreshold,
            "k": self.thresholding_tab.kValue
        }
        for widget in self.thresholding_ui.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.update_params("thresholding", self.thresholding_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.update_params("thresholding", self.thresholding_ui))

        # Frequency Filter Tab
        self.frequency_filter_ui = {
            "filter_type": self.frequency_filter_tab.freqType,
            "radius": self.frequency_filter_tab.freqRadius
        }
        for widget in self.frequency_filter_ui.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.update_params("frequency_filter", self.frequency_filter_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.update_params("frequency_filter", self.frequency_filter_ui))

        # Hybrid Image Tab
        self.hybrid_image_ui = {
            "cutoff1": self.hybrid_image_tab.cutoff1,
            "cutoff2": self.hybrid_image_tab.cutoff2,
            "type1": self.hybrid_image_tab.type1,
            "type2": self.hybrid_image_tab.type2
        }
        for widget in self.hybrid_image_ui.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.update_params("hybrid_image", self.hybrid_image_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.update_params("hybrid_image", self.hybrid_image_ui))    
        # Connect apply buttons
        self.noise_filter_tab.btn_noise.clicked.connect(self.apply_noise)
           # Connect the "Apply Filter" button
        self.noise_filter_tab.btn_filter.clicked.connect(self.apply_filter)
        # self.edge_detection_tab.btn_edge_detection.clicked.connect(lambda: self.process_image("detect_edges", **self.params["edge_detection"]))
        # self.thresholding_tab.btn_threshold.clicked.connect(lambda: self.process_image("apply_thresholding", **self.params["thresholding"]))
        # self.frequency_filter_tab.btn_freq_filter.clicked.connect(lambda: self.process_image("apply_frequency_filter", **self.params["frequency_filter"]))
        # self.hybrid_image_tab.btn_hybrid.clicked.connect(lambda: self.process_image("create_hybrid_image", **self.params["hybrid_image"]))
    
    def update_params(self, tab_name, ui_components):
        """
        Update the parameters for a specific tab based on the UI components.
        
        Args:
            tab_name (str): The name of the tab (e.g., "noise_filter").
            ui_components (dict): A dictionary of UI components and their keys.
        """
        print("Updating params for", tab_name)
        self.params[tab_name] = {}
        for key, widget in ui_components.items():
            if isinstance(widget, (QComboBox, QLineEdit)):
                self.params[tab_name][key] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self.params[tab_name][key] = widget.value()
            elif isinstance(widget, QCheckBox):
                self.params[tab_name][key] = widget.isChecked()
        
        print(self.params[tab_name])        

    def on_image_label_double_click(self, event):
        self.load_image()
    

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap.scaled(self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio))

    def apply_noise(self):
        """
        Applies noise to the image based on the selected noise type and parameters from the UI.
        """
        # Retrieve noise parameters from the params dictionary
        noise_params = self.params["noise_filter"]

        # Call the add_noise function with the retrieved parameters
        self._add_noise(**noise_params)
        print("Applying noise:", noise_params)
        self.display_image(self.modified_image)
        
    def _add_noise(self, **kwargs):
        """
        Adds noise to the image based on the specified noise type and parameters.

        Args:
            noise_type (str): Type of noise to add. Options: "uniform", "gaussian", "salt_pepper".
            **kwargs: Additional parameters for the noise (e.g., intensity, mean, std, salt_prob, pepper_prob).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying new noise

        if self.image is not None:
            # Call the noise processor with the specified noise type and parameters
            noisy_image = self.processors['noise'].add_noise(**kwargs)
            self.modified_image = noisy_image
            self.display_image(self.modified_image, modified=True)
        else:
            raise ValueError("No image loaded. Please load an image before applying noise.")
            
    def apply_filter(self):
        """
        Applies a filter to the image based on the selected filter type and parameters from the UI.
        """
        # Retrieve filter parameters from the params dictionary
        filter_params = self.params.get("filtering", {})
        # Call the apply_filters function with the retrieved parameters
        self._apply_filters(**filter_params)
        print("Applying filter:", filter_params)
        self.display_image(self.modified_image)

    def _apply_filters(self, **kwargs):
        """
        Applies a filter to the image based on the specified filter type and parameters.

        Args:
            filter_type (str): Type of filter to apply. Options: "average", "gaussian", "median".
            **kwargs: Additional parameters for the filter (e.g., kernel_size, sigma).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying a new filter

        if self.image is not None:
            # Apply the filter using the specified parameters
            filtered_image = self.processors['noise'].apply_filters( **kwargs)
            self.modified_image = filtered_image.get(kwargs.get("filter_type", "median")) # Default to "median" filter
        else:
            raise ValueError("No image loaded. Please load an image before applying a filter.")
    def detect_edges(self):
        """
        Detects edges in the image based on the selected edge detection type and parameters from the UI.
        """
        # Retrieve edge detection parameters from the params dictionary
        edge_params = self.params.get("edge_detection", {})

        # Call the _detect_edges function with the retrieved parameters
        self._detect_edges(**edge_params)
        print("Detecting edges:", edge_params)
        
        self.display_image(self.modified_image, modified=True)

    def _detect_edges(self, **kwargs):
        """ 
        Detects edges in the image based on the specified edge detection type and parameters.

        Args:
            edge_type (str): Type of edge detection to apply. Options: "sobel", "canny", "prewitt", "roberts".
            **kwargs: Additional parameters for the edge detection (e.g., kernel_size, sigma, thresholds).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying edge detection

        if self.image is not None:
            # Apply edge detection using the specified parameters
            edge_map = self.processors['edge_detector'].detect_edges(**kwargs)
            self.modified_image = edge_map
        else:
            raise ValueError("No image loaded. Please load an image before detecting edges.")
    
    def apply_thresholding(self):
        """
        Applies thresholding to the image based on the selected thresholding type and parameters from the UI.
        """
        # Retrieve thresholding parameters from the params dictionary
        threshold_params = self.params.get("thresholding", {})
        # Call the _apply_thresholding function with the retrieved parameters
        print("Applying thresholding:", threshold_params)

        self._apply_thresholding( **threshold_params)
        self.display_image(self.modified_image, modified=True)
        

    def _apply_thresholding(self,  **kwargs):
        """
        Applies thresholding to the image based on the specified thresholding type and parameters.

        Args:
            threshold_type (str): Type of thresholding to apply. Options: "global", "local".
            **kwargs: Additional parameters for the thresholding (e.g., threshold_value, kernel_size, k_value).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying thresholding

        if self.image is not None:
            # Apply thresholding using the specified parameters
            thresholded_image = self.processors['thresholding'].apply_thresholding(**kwargs)
            self.modified_image = thresholded_image
        else:
            raise ValueError("No image loaded. Please load an image before applying thresholding.")
    
    def show_histogram(self):
        
        # self.processors['histogram'].plot_all_histograms()
        self.processors['histogram'].set_image(self.image)

        self.visualization_widget = HistogramVisualizationWidget(processor  = self.processors['histogram'])
        self.visualization_widget.show()
    
    def apply_frequency_filter(self):
        """
        Applies a frequency filter to the image based on the selected filter type and parameters from the UI.
        """
        # Retrieve frequency filter parameters from the params dictionary
        frequency_params = self.params.get("frequency_filter", {})
        # Call the _apply_frequency_filter function with the retrieved parameters
        self._apply_frequency_filter(**frequency_params)
        print("Applying frequency filter:", frequency_params)
        self.display_image(self.modified_image, modified=True)
        

    def _apply_frequency_filter(self, **kwargs):
        """
        Applies a frequency filter to the image based on the specified filter type and parameters.

        Args:
            filter_type (str): Type of frequency filter to apply. Options: "low_pass", "high_pass".
            **kwargs: Additional parameters for the frequency filter (e.g., radius).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying frequency filtering

        if self.image is not None:
            # Apply frequency filter using the specified parameters
            filtered_image = self.processors['frequency'].apply_filter(**kwargs)
            self.modified_image = filtered_image
        else:
            raise ValueError("No image loaded. Please load an image before applying frequency filtering.")
   
    def equalize(self):
        self.modified_image = self.processors['image'].get_equalized_image()
        self.display_image(self.modified_image)

    def normalize(self):
        self.modified_image = self.processors['image'].get_normalized_image()
        self.display_image(self.modified_image)

    def load_image(self, hybird = False):
        """
        Load an image from disk and display it in the UI.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path and hybird == False:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.original_image = self.image
            if self.image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.display_image(self.image)
        elif hybird == True:
            self.extra_image = cv2.imread(file_path)
            if self.extra_image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return

            self.display_image(self.extra_image, hybird = True)
        else:
            QMessageBox.information(self, "Info", "No file selected.")

    def display_image(self, img, hybrid=False, modified=False):
        """
        Convert a NumPy BGR image to QImage and display it in lbl_image.
        """
        if len(img.shape) == 3:
            # Convert BGR to RGB
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale
            h, w = img.shape
            # Ensure the image is in uint8 format
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # Convert the NumPy array to bytes
            img_bytes = img.tobytes()
            qimg = QImage(img_bytes, w, h, w, QImage.Format_Indexed8)
        
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap.scaled(
            self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio
        ))
    

    def equalize(self):
        """
        Example: Using the factory to create a HistogramProcessor.
        """
        if self.modified_image is not None:
            self.confirm_edit()
            
        if self.image is not None:
            equalized_image = self.processors['image'].get_equalized_image() 
            self.modified_image = equalized_image
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No image available. Load an image first.")
    def normalize(self):
        """
        Example: Using the factory to create a HistogramProcessor.
        """
        if self.modified_image is not None:
            self.confirm_edit()
        if self.image is not None:
            normalized_image = self.processors['image'].get_normalized_image() 
            self.modified_image = normalized_image
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No image available. Load an image first.")
    
    def confirm_edit(self):
        """
        Confirm the edit.
        """
        if self.modified_image is not None:
            self.image = self.modified_image
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No image available. Load an image first.")
    
    def discard_edit(self):
        """
        Discard the edit.
        """
        if self.modified_image is not None:
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No image available. Load an image first.")
    def reset_image(self):
        """
        Reset the image to the original.
        """
        if self.original_image is not None:
            self.image = self.original_image
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No original image available. Load an image first.")

    def detect_lines(self):
        """
        Detects lines in the image using the Hough Transform based on the selected parameters from the UI.
        """
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        shape_params = self.params["shape_detection"]
    
        
        self.modified_image = self.processors['edge_detector'].detect_shape(
            shape_type = 'line',
            **shape_params
        )

        self.display_image(self.modified_image)

    def detect_circles(self):
        """
        Detects lines in the image using the Hough Transform based on the selected parameters from the UI.
        """
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        shape_params = self.params["shape_detection"]

        
        self.modified_image = self.processors['edge_detector'].detect_shape(
            shape_type = 'circle',
            **shape_params
        )

        self.display_image(self.modified_image)

    # def detect_circles(self):
    #     """
    #     Detects circles in the image using the Hough Transform based on the selected parameters from the UI.
    #     """
    #     if self.image is None:
    #         QMessageBox.warning(self, "Warning", "Please load an image first.")
    #         return

    #     try:
    #         min_edge_threshold = self.hough_transform_tab.minEdgeThreshold.value()
    #         max_edge_threshold = self.hough_transform_tab.maxEdgeThreshold.value()
    #         r_min = self.hough_transform_tab.rMin.value()
    #         r_max = self.hough_transform_tab.rMax.value()
    #         delta_r = self.hough_transform_tab.deltaR.value()
    #         num_thetas = self.hough_transform_tab.numThetas.value()
    #         bin_threshold = self.hough_transform_tab.binThreshold.value()

    #         self.modified_image = detect_circles(
    #             self.image,
    #             min_edge_threshold=min_edge_threshold,
    #             max_edge_threshold=max_edge_threshold,
    #             r_min=r_min,
    #             r_max=r_max,
    #             delta_r=delta_r,
    #             num_thetas=num_thetas,
    #             bin_threshold=bin_threshold
    #         )

    #         self.display_image(self.modified_image)
    #     except Exception as e:
    #         print(f"Error in detect_circles: {e}")
    #         QMessageBox.critical(self, "Error", f"Failed to detect circles: {e}")

def main():
    app = QApplication(sys.argv)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qss_path = os.path.join(script_dir, "../resources/styles.qss")
    
    with open(qss_path, "r") as file:
        app.setStyleSheet(file.read())
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

