import sys
import cv2
import numpy as np
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, 
    QVBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from processor_factory import ProcessorFactory
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QFrame,QTabWidget,QSpacerItem,QSizePolicy,
    QVBoxLayout, QWidget, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout
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
                self.display_image(self.image1, self.image1_label)
            elif image_number == 2:
                self.image2 = image
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
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        #? Left Frame with TabWidget
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

        control_frame=QFrame()
        control_frame.setMaximumHeight(100)
        control_layout = QHBoxLayout(control_frame)

        right_layout = QVBoxLayout(right_frame)
        right_layout.setAlignment(Qt.AlignHCenter)

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

        control_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

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

        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        image_display_layout.addWidget(self.btn_load_image)

        right_layout.addWidget(image_display_frame)
        main_layout.addWidget(right_frame)

        # Image & Processor Variables
        self.image = None
        self.original_image = None
        self.modified_image = None
        self.extra_image = None
        self.processors = {key: ProcessorFactory.create_processor(key) for key in ['noise', 'edge_detector', 'thresholding', 'frequency', 'histogram', 'image']}

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
        noise_type = self.noise_filter_tab.noiseType.currentText()
        self.modified_image = self.processors['noise'].add_noise(noise_type)
        self.display_image(self.modified_image)

    def apply_filter(self):
        filter_type = self.noise_filter_tab.filterType.currentText()
        kernel_size = self.noise_filter_tab.kernelSize.value()
        sigma = self.noise_filter_tab.sigmaValue.value()
        self.modified_image = self.processors['noise'].apply_filters(filter_type, kernel_size=kernel_size, sigma=sigma)
        self.display_image(self.modified_image)

    def detect_edges(self):
        edge_type = self.edge_detection_tab.edgeType.currentText()
        self.modified_image = self.processors['edge_detector'].detect_edges(edge_type)
        self.display_image(self.modified_image)

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

    def add_noise(self, noise_type="uniform", **kwargs):
        """
        Example: Using the factory to create a NoiseProcessor.
        
        """
        if self.modified_image is not None:
            self.confirm_edit()
                    
        if self.image is not None:
            noisy_image = self.processors['noise'].add_noise(noise_type, **kwargs) #uniform (intensity=50), gaussian (mean=0, std=25), salt_pepper (salt_prob=0.02, pepper_prob=0.02)
            self.modified_image = noisy_image
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError(f"Unknown noise type '{noise_type}'. Use 'gaussian' or 'salt_pepper'.")
    
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
    
    def detect_edges(self, edge_type="sobel", **kwargs):
        """
        Example: Using the factory to create an EdgeDetector.
        """ 
        if self.modified_image is not None:
            self.confirm_edit()
        
        if self.image is not None:
            edge_map = self.processors['edge_detector'].detect_edges(edge_type, **kwargs)
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

