import sys
import cv2
import numpy as np
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QTabWidget, QComboBox, QSpinBox, QDoubleSpinBox, 
    QPushButton, QLabel, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from processor_factory import ProcessorFactory
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QFrame,QTabWidget,QSpacerItem,QSizePolicy,
    QVBoxLayout, QWidget, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from processor_factory import ProcessorFactory

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

        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity"))
        intensity_layout.addWidget(self.noiseIntensity)
        self.noiseParamLayout.addLayout(intensity_layout)

        mean_layout = QHBoxLayout()
        mean_layout.addWidget(QLabel("Mean"))
        mean_layout.addWidget(self.gaussianMean)
        self.noiseParamLayout.addLayout(mean_layout)

        std_layout = QHBoxLayout()
        std_layout.addWidget(QLabel("Std Dev"))
        std_layout.addWidget(self.gaussianStd)
        self.noiseParamLayout.addLayout(std_layout)

        salt_layout = QHBoxLayout()
        salt_layout.addWidget(QLabel("Salt Prob"))
        salt_layout.addWidget(self.saltProb)
        self.noiseParamLayout.addLayout(salt_layout)

        pepper_layout = QHBoxLayout()
        pepper_layout.addWidget(QLabel("Pepper Prob"))
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

class EdgeDetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        edge_detection_frame = QFrame()
        edge_detection_frame.setObjectName("edge_detection_frame")
        edge_detection_layout = QVBoxLayout(edge_detection_frame)
        edge_detection_layout.setAlignment(Qt.AlignTop)

        self.edgeType = QComboBox()
        self.edgeType.addItems(["sobel", "canny", "prewitt", "roberts"])

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

        sobel_kernel_layout = QHBoxLayout()
        sobel_kernel_layout.addWidget(QLabel("Kernel Size"))
        sobel_kernel_layout.addWidget(self.sobelKernelSize)
        self.edgeParamLayout.addLayout(sobel_kernel_layout)

        sobel_sigma_layout = QHBoxLayout()
        sobel_sigma_layout.addWidget(QLabel("Sigma"))
        sobel_sigma_layout.addWidget(self.sobelSigma)
        self.edgeParamLayout.addLayout(sobel_sigma_layout)

        canny_low_layout = QHBoxLayout()
        canny_low_layout.addWidget(QLabel("Low Threshold"))
        canny_low_layout.addWidget(self.cannyLowThreshold)
        self.edgeParamLayout.addLayout(canny_low_layout)

        canny_high_layout = QHBoxLayout()
        canny_high_layout.addWidget(QLabel("High Threshold"))
        canny_high_layout.addWidget(self.cannyHighThreshold)
        self.edgeParamLayout.addLayout(canny_high_layout)

        canny_max_layout = QHBoxLayout()
        canny_max_layout.addWidget(QLabel("Max Edge"))
        canny_max_layout.addWidget(self.cannyMaxEdgeVal)
        self.edgeParamLayout.addLayout(canny_max_layout)

        canny_min_layout = QHBoxLayout()
        canny_min_layout.addWidget(QLabel("Min Edge"))
        canny_min_layout.addWidget(self.cannyMinEdgeVal)
        self.edgeParamLayout.addLayout(canny_min_layout)

        prewitt_threshold_layout = QHBoxLayout()
        prewitt_threshold_layout.addWidget(QLabel("Threshold"))
        prewitt_threshold_layout.addWidget(self.prewittThreshold)
        self.edgeParamLayout.addLayout(prewitt_threshold_layout)

        prewitt_value_layout = QHBoxLayout()
        prewitt_value_layout.addWidget(QLabel("Value"))
        prewitt_value_layout.addWidget(self.prewittValue)
        self.edgeParamLayout.addLayout(prewitt_value_layout)

        self.edgeParamLayout.addWidget(self.btn_edge_detection)

        edge_detection_layout.addLayout(self.edgeParamLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(edge_detection_frame)

class ThresholdingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        thresholding_frame = QFrame()
        thresholding_frame.setObjectName("thresholding_frame")
        thresholding_layout = QVBoxLayout(thresholding_frame)
        thresholding_layout.setAlignment(Qt.AlignTop)

        self.thresholdType = QComboBox()
        self.thresholdType.addItems(["global", "local"])

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

        global_threshold_layout = QHBoxLayout()
        global_threshold_layout.addWidget(QLabel("Global Threshold"))
        global_threshold_layout.addWidget(self.globalThreshold)
        self.thresholdLayout.addLayout(global_threshold_layout)

        kernel_size_threshold_layout = QHBoxLayout()
        kernel_size_threshold_layout.addWidget(QLabel("Kernel Size"))
        kernel_size_threshold_layout.addWidget(self.kernelSizeThreshold)
        self.thresholdLayout.addLayout(kernel_size_threshold_layout)

        k_value_layout = QHBoxLayout()
        k_value_layout.addWidget(QLabel("K Value"))
        k_value_layout.addWidget(self.kValue)
        self.thresholdLayout.addLayout(k_value_layout)

        self.thresholdLayout.addWidget(self.btn_threshold)

        thresholding_layout.addLayout(self.thresholdLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(thresholding_frame)

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
        
        hybrid_image_frame = QFrame()
        hybrid_image_frame.setObjectName("hybrid_image_frame")
        hybrid_image_layout = QVBoxLayout(hybrid_image_frame)
        hybrid_image_layout.setAlignment(Qt.AlignTop)

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
        self.btn_hybrid.clicked.connect(parent.create_hybrid_image)

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

        hybrid_image_layout.addLayout(self.hybridLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(hybrid_image_frame)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Image Processing App")
        self.setGeometry(100, 100, 1200, 800)  # Set window size

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout()
        central_widget.setLayout(self.main_layout)
        
        self.init_ui(self.main_layout)

        # self.noise_filter_tab = None
        # self.edge_detection_tab = None
        # self.thresholding_tab = None
        # self.frequency_filter_tab = None
        # self.hybrid_image_tab = None
        
        # Single data structure to store all parameters
        self.params = {
            "noise_filter": {},
            "filtering": {},
            "edge_detection": {},
            "thresholding": {},
            "frequency_filter": {},
            "hybrid_image": {}
        }
        
        self.connect_signals()
        # Image & Processor Variables
        self.image = None
        self.original_image = None
        self.modified_image = None
        self.extra_image = None
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

        self.btn_histogram = QPushButton("Show Histogram")
        self.btn_histogram.clicked.connect(self.show_histogram)
        control_buttons_layout.addWidget(self.btn_histogram)

        self.btn_equalize = QPushButton("Equalize Image")
        self.btn_equalize.clicked.connect(self.equalize)
        control_buttons_layout.addWidget(self.btn_equalize)

        self.btn_normalize = QPushButton("Normalize Image")
        self.btn_normalize.clicked.connect(self.normalize)
        control_buttons_layout.addWidget(self.btn_normalize)

        self.btn_snake = QPushButton("Active Contour (Snake)")
        # self.btn_snake.clicked.connect(self.run_snake)
        control_buttons_layout.addWidget(self.btn_snake)

        control_layout.addWidget(control_buttons_frame)

        # Add a spacer to push the next frame to the right
        control_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Image Control Buttons Frame
        image_control_buttons_frame = QFrame()
        image_control_buttons_layout = QHBoxLayout(image_control_buttons_frame)

        self.btn_confirm = QPushButton("Confirm Edit")
        self.btn_confirm.clicked.connect(self.confirm_edit)
        image_control_buttons_layout.addWidget(self.btn_confirm)

        self.btn_discard = QPushButton("Discard Edit")
        self.btn_discard.clicked.connect(self.discard_edit)
        image_control_buttons_layout.addWidget(self.btn_discard)

        self.btn_reset = QPushButton("Reset Image")
        self.btn_reset.clicked.connect(self.reset_image)
        image_control_buttons_layout.addWidget(self.btn_reset)
        control_layout.addWidget(image_control_buttons_frame)

        # Add the control frame to the right layout
        right_layout.addWidget(control_frame)

        # Image Display Frame
        image_display_frame = QFrame()
        image_display_frame.setMaximumWidth(1400)
        image_display_frame.setMinimumWidth(1400)
        image_display_layout = QVBoxLayout(image_display_frame)

        self.lbl_image = QLabel("No Image Loaded")
        self.lbl_image.setObjectName("lbl_image")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        image_display_layout.addWidget(self.lbl_image)

        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        image_display_layout.addWidget(self.btn_load_image)

        # Add the image display frame to the right layout
        right_layout.addWidget(image_display_frame)

        # Add the right frame to the main layout
        main_layout.addWidget(right_frame)
    
    def connect_signals(self):
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
            "edge_type": self.edge_detection_tab.edgeType,
            "sobel_kernel_size": self.edge_detection_tab.sobelKernelSize,
            "sobel_sigma": self.edge_detection_tab.sobelSigma,
            "canny_low_threshold": self.edge_detection_tab.cannyLowThreshold,
            "canny_high_threshold": self.edge_detection_tab.cannyHighThreshold,
            "canny_max_edge_val": self.edge_detection_tab.cannyMaxEdgeVal,
            "canny_min_edge_val": self.edge_detection_tab.cannyMinEdgeVal,
            "prewitt_threshold": self.edge_detection_tab.prewittThreshold,
            "prewitt_value": self.edge_detection_tab.prewittValue
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
            "global_threshold": self.thresholding_tab.globalThreshold,
            "kernel_size": self.thresholding_tab.kernelSizeThreshold,
            "k_value": self.thresholding_tab.kValue
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

        

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.original_image = self.image.copy()
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.display_image(self.image)

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
        # noise_type = noise_params.get("noise_type", "uniform")  # Default to "uniform" if not specified

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
        edge_type = edge_params.get("edge_type", "sobel")  # Default to "sobel" if not specified

        # Call the _detect_edges function with the retrieved parameters
        self._detect_edges(**edge_params)
        print("Detecting edges:", edge_params)
        
        self.display_image(self.modified_image, modified=True)

    
    def _detect_edges(self, edge_type="sobel", **kwargs):
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
        threshold_type = self.thresholding_tab.thresholdType.currentText()
        self.modified_image = self.processors['thresholding'].apply_threshold(threshold_type)
        self.display_image(self.modified_image)

    def show_histogram(self):
        self.processors['histogram'].plot_all_histograms()

    def apply_frequency_filter(self):
        self.modified_image = self.processors['frequency'].apply_filter("low_pass")
        self.display_image(self.modified_image)

    def create_hybrid_image(self):
        self.modified_image = self.processors['frequency'].create_hybrid_image(self.image, self.extra_image)
        self.display_image(self.modified_image)

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

    def display_image(self, img, hybird = False, modified = False):
        """
        Convert a NumPy BGR image to QImage and display it in lbl_image.
        """
        if len(img.shape) == 3:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Indexed8)
        
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap.scaled(self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio))

   
    def apply_filters(self, filter_type="median", **kwargs):
        """
        Example: Using the factory  to create a NoiseProcessor.
        """
        if self.modified_image is not None:
            self.confirm_edit()
        if self.modified_image is not None:
            filtered_image = self.processors['noise'].apply_filters() #average (kernel_size=3), gaussian (kernel_size=3, sigma=1.0), median (kernel_size=3)
            self.modified_image = filtered_image[filter_type]
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No noisy image available. Apply noise first.")        
     
    def apply_thresholding(self, threshold_type="global", **kwargs):
        """
        Example: Using the factory to create a ThresholdingProcessor.
        """ 
        if self.modified_image is not None:
            self.confirm_edit()
            
        if self.modified_image is not None:
            thresholded_image = self.processors['thresholding'].apply_threshold(threshold_type, **kwargs) # threshold_type : (local:  kernel=4, k=2), (global: T=128)
            self.modified_image = thresholded_image
            self.display_image(self.modified_image, modified = True)
        else:    
            raise ValueError("No image available. Load an image first.")
    
    def detect_edges(self, **kwargs):
        """
        Example: Using the factory to create an EdgeDetector.
        """ 
        if self.modified_image is not None:
            self.confirm_edit()
        
        if self.image is not None:
            edge_map = self.processors['edge_detector'].detect_edges(**kwargs)
            # sobel (kernal_size=3, sigma=1.0), canny (low_threshold=50, high_threshold=150, max_edge_val=255, min_edge_val=0), prewitt (threshold=50, value=255), roberts
            self.modified_image = edge_map 
            self.display_image(self.modified_image, modified = True)
        else:    
            raise ValueError("No image available. Load an image first.")     

    def show_histograms_with_CDF(self, **kwargs):
        """
        Example: Using the factory to create a HistogramProcessor.
        """
           
        if self.image is not None:
            self.processors['histogram'].plot_all_histograms(**kwargs) # grayscale, rgb
        else:
            raise ValueError("No image available. Load an image first.")
    
    def apply_frequency_filter(self, filter_type="low_pass", **kwargs):
        """
        Example: Using the factory to create a FrequencyProcessor.
        """ 
        if self.modified_image is not None:
            self.confirm_edit()
            
        if self.image is not None:
            filtered_image = self.processors['frequency'].apply_filter(filter_type, **kwargs) # low_pass (radius=10), high_pass (radius=10)
            self.modified_image = filtered_image    
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No image available. Load an image first.")
    def hybrid_image(self, **kwargs):
        """
        Example: Using the factory to create a HybridProcessor.
        """
        if self.modified_image is not None:
            self.confirm_edit()
            
        self.load_image(hybird=True)
        
        if self.image is not None:
            hybrid_image = self.processors['frequency'].create_hybrid_image(img2 = self.extra_image,**kwargs) # cutoff1=10, cutoff2=10, type1="lp", type2="hp"
            self.modified_image = hybrid_image
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No image available. Load an image first.")

    # def noise_and_filter(self):
        """
        Demonstrate noise addition + filtering.
        """
        
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return

        # Use factory to create a noise processor
        noise_proc = ProcessorFactory.create_processor("noise", self.image)
        # Add Gaussian noise
        noisy_img = noise_proc.add_gaussian_noise(mean=0, std=25)

        # Then filter the noisy image
        noise_proc.set_image(noisy_img)
        filtered_imgs = noise_proc.apply_filters()  # average, gauss, median
        # Let's just display the median filter result
        median_filtered = filtered_imgs["Median Filter"]
        
        # Convert to BGR for display
        bgr_display = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
        self.display_image(bgr_display)
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

