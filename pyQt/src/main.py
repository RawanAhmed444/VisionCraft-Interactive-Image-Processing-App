import sys
import cv2
import numpy as np
import os
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize

from processor_factory import ProcessorFactory
from classes.histogram_processor import HistogramVisualizationWidget
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QFrame,QTabWidget,QSpacerItem,QSizePolicy,
    QVBoxLayout, QWidget, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QCheckBox, QGroupBox
)

from processor_factory import ProcessorFactory
from functions.hough_transform_functions import detect_lines,detect_circles

class HoughTransformTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        hough_transform_layout = QVBoxLayout(self)

        hough_frame = QFrame()
        hough_frame.setObjectName("hough_frame")
        hough_layout = QVBoxLayout(hough_frame)

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

        self.blurKSize = QSpinBox()
        self.blurKSize.setRange(1, 15)
        self.blurKSize.setValue(5)

        self.lowThreshold = QSpinBox()
        self.lowThreshold.setRange(0, 255)
        self.lowThreshold.setValue(50)

        self.highThreshold = QSpinBox()
        self.highThreshold.setRange(0, 255)
        self.highThreshold.setValue(150)

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

        blur_ksize_layout = QHBoxLayout()
        blur_ksize_layout.addWidget(QLabel("Blur Kernel Size"))
        blur_ksize_layout.addWidget(self.blurKSize)
        line_layout.addLayout(blur_ksize_layout)

        low_threshold_layout = QHBoxLayout()
        low_threshold_layout.addWidget(QLabel("Low Threshold"))
        low_threshold_layout.addWidget(self.lowThreshold)
        line_layout.addLayout(low_threshold_layout)

        high_threshold_layout = QHBoxLayout()
        high_threshold_layout.addWidget(QLabel("High Threshold"))
        high_threshold_layout.addWidget(self.highThreshold)
        line_layout.addLayout(high_threshold_layout)

        hough_threshold_ratio_layout = QHBoxLayout()
        hough_threshold_ratio_layout.addWidget(QLabel("Hough Threshold Ratio"))
        hough_threshold_ratio_layout.addWidget(self.houghThresholdRatio)
        line_layout.addLayout(hough_threshold_ratio_layout)

        line_layout.addWidget(self.btn_detect_lines)

        ############################## Circle Detection Parameters ##############################
        circle_group_frame = QFrame()
        circle_group_frame.setObjectName("circle_group_frame")
        circle_layout = QVBoxLayout(circle_group_frame)

        self.minEdgeThreshold = QSpinBox()
        self.minEdgeThreshold.setRange(0, 255)
        self.minEdgeThreshold.setValue(50)

        self.maxEdgeThreshold = QSpinBox()
        self.maxEdgeThreshold.setRange(0, 255)
        self.maxEdgeThreshold.setValue(150)

        self.rMin = QSpinBox()
        self.rMin.setRange(1, 100)
        self.rMin.setValue(20)

        self.rMax = QSpinBox()
        self.rMax.setRange(1, 500)
        self.rMax.setValue(100)

        self.deltaR = QSpinBox()
        self.deltaR.setRange(1, 10)
        self.deltaR.setValue(1)

        self.numThetas = QSpinBox()
        self.numThetas.setRange(1, 360)
        self.numThetas.setValue(50)

        self.binThreshold = QDoubleSpinBox()
        self.binThreshold.setRange(0.0, 1.0)
        self.binThreshold.setSingleStep(0.1)
        self.binThreshold.setValue(0.4)

        self.btn_detect_circles = QPushButton("Detect Circles")
        self.btn_detect_circles.clicked.connect(parent.detect_circles)

        min_edge_threshold_layout = QHBoxLayout()
        min_edge_threshold_layout.addWidget(QLabel("Min Edge Threshold"))
        min_edge_threshold_layout.addWidget(self.minEdgeThreshold)
        circle_layout.addLayout(min_edge_threshold_layout)

        max_edge_threshold_layout = QHBoxLayout()
        max_edge_threshold_layout.addWidget(QLabel("Max Edge Threshold"))
        max_edge_threshold_layout.addWidget(self.maxEdgeThreshold)
        circle_layout.addLayout(max_edge_threshold_layout)

        r_min_layout = QHBoxLayout()
        r_min_layout.addWidget(QLabel("Min Radius"))
        r_min_layout.addWidget(self.rMin)
        circle_layout.addLayout(r_min_layout)

        r_max_layout = QHBoxLayout()
        r_max_layout.addWidget(QLabel("Max Radius"))
        r_max_layout.addWidget(self.rMax)
        circle_layout.addLayout(r_max_layout)

        delta_r_layout = QHBoxLayout()
        delta_r_layout.addWidget(QLabel("Delta Radius"))
        delta_r_layout.addWidget(self.deltaR)
        circle_layout.addLayout(delta_r_layout)

        num_thetas_layout = QHBoxLayout()
        num_thetas_layout.addWidget(QLabel("Num Thetas"))
        num_thetas_layout.addWidget(self.numThetas)
        circle_layout.addLayout(num_thetas_layout)

        bin_threshold_layout = QHBoxLayout()
        bin_threshold_layout.addWidget(QLabel("Bin Threshold"))
        bin_threshold_layout.addWidget(self.binThreshold)
        circle_layout.addLayout(bin_threshold_layout)

        circle_layout.addWidget(self.btn_detect_circles)

        hough_layout.addWidget(line_group_frame)
        hough_layout.addWidget(circle_group_frame)
        hough_transform_layout.addWidget(hough_frame)


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
            "shape_detection":{}
            
        }
        
        self.connect_signals()
        # Image & Processor Variables
        self.image = None
        self.original_image = None
        self.modified_image = None
        self.processors = {key: ProcessorFactory.create_processor(key) for key in ['noise', 'edge_detector', 'thresholding', 'frequency', 'histogram', 'image']}


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

        
        left_layout.addWidget(tab_widget)
        main_layout.addWidget(left_frame)
        
        #? Right Frame with Control Buttons and Image Display
        right_frame = QFrame()
        right_frame.setObjectName("right_frame")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setAlignment(Qt.AlignHCenter)
        
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

        #we will edit this part
        self.btn_snake = QPushButton()
        self.btn_snake.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/contour-map.png')))
        self.btn_snake.setIconSize(QSize(32, 32))
        # self.btn_snake.clicked.connect(self.run_snake)
        control_buttons_layout.addWidget(self.btn_snake)

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
        image_display_frame.setMaximumWidth(1390)
        image_display_frame.setMinimumWidth(1390)
        image_display_layout = QVBoxLayout(image_display_frame)

        self.lbl_image = QLabel("No Image Loaded")
        self.lbl_image.setObjectName("lbl_image")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        image_display_layout.addWidget(self.lbl_image)
        self.lbl_image.mouseDoubleClickEvent = self.on_image_label_double_click


        # Add the image display frame to the right layout
        right_layout.addWidget(image_display_frame)

        # Add the right frame to the main layout
        main_layout.addWidget(right_frame)
    
    def connect_signals(self):
        
        #hough_transform and shape detection tab
        shape_detection_ui = {
            'num_rho' : self.hough_transform_tab.numRho,
            'num_theta': self.hough_transform_tab.numTheta,
            'blur_ksize': self.hough_transform_tab.blurKSize,
            'hough_threshold_ratio':self.hough_transform_tab.houghThresholdRatio,
            'r_min' : self.hough_transform_tab.rMin,
            'r_max': self.hough_transform_tab.rMax,
            'num_thetas' : self.hough_transform_tab.numThetas         
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

