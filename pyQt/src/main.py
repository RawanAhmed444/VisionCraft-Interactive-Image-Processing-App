import sys
import cv2
import numpy as np

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
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, 
    QVBoxLayout, QWidget, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from processor_factory import ProcessorFactory

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Image Processing App")
        self.setGeometry(100, 100, 800, 600)  # Set window size

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)

        # Image Display
        self.lbl_image = QLabel("No Image Loaded")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 2px solid black;")
        self.layout.addWidget(self.lbl_image)

        # Load Image Button
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        self.layout.addWidget(self.btn_load_image)

                # Noise Section UI Components
        self.noiseType = QComboBox()
        self.noiseType.addItems(["uniform", "gaussian", "salt_pepper"])
        # self.noiseType.currentIndexChanged.connect(self.update_noise_layout)

        # Uniform Noise
        self.noiseIntensity = QSpinBox()
        self.noiseIntensity.setRange(1, 100)
        self.noiseIntensity.setValue(50)

        # Gaussian Noise
        self.gaussianMean = QDoubleSpinBox()
        self.gaussianMean.setRange(-50, 50)
        self.gaussianMean.setValue(0)

        self.gaussianStd = QDoubleSpinBox()
        self.gaussianStd.setRange(1, 100)
        self.gaussianStd.setValue(25)

        # Salt & Pepper Noise
        self.saltProb = QDoubleSpinBox()
        self.saltProb.setRange(0.0, 1.0)
        self.saltProb.setSingleStep(0.01)
        self.saltProb.setValue(0.02)

        self.pepperProb = QDoubleSpinBox()
        self.pepperProb.setRange(0.0, 1.0)
        self.pepperProb.setSingleStep(0.01)
        self.pepperProb.setValue(0.02)

        # Noise Button
        self.btn_noise = QPushButton("Add Noise")
        self.btn_noise.clicked.connect(self.apply_noise)

        # Layout for Noise Parameters
        self.noiseParamLayout = QHBoxLayout()
        self.noiseParamLayout.addWidget(self.noiseType)
        self.noiseParamLayout.addWidget(self.noiseIntensity)  # Default (Uniform)
        self.noiseParamLayout.addWidget(self.btn_noise)
        self.layout.addLayout(self.noiseParamLayout)
        
        
        # Filter Section
        self.filterType = QComboBox()
        self.filterType.addItems(["average", "gaussian", "median"])
        self.kernelSize = QSpinBox()
        self.kernelSize.setRange(1, 15)
        self.kernelSize.setValue(3)
        self.sigmaValue = QDoubleSpinBox()
        self.sigmaValue.setRange(0.1, 10.0)
        self.sigmaValue.setValue(1.0)
        self.btn_filter = QPushButton("Apply Filter")
        self.btn_filter.clicked.connect(self.apply_filter)

        filterLayout = QHBoxLayout()
        filterLayout.addWidget(self.filterType)
        filterLayout.addWidget(self.kernelSize)
        filterLayout.addWidget(self.sigmaValue)
        filterLayout.addWidget(self.btn_filter)
        self.layout.addLayout(filterLayout)

        # Edge Detection Section UI Components
        self.edgeType = QComboBox()
        self.edgeType.addItems(["sobel", "canny", "prewitt", "roberts"])
        # self.edgeType.currentIndexChanged.connect(self.update_edge_layout)

        # Sobel Parameters
        self.sobelKernelSize = QSpinBox()
        self.sobelKernelSize.setRange(1, 15)
        self.sobelKernelSize.setValue(3)

        self.sobelSigma = QDoubleSpinBox()
        self.sobelSigma.setRange(0.1, 10.0)
        self.sobelSigma.setValue(1.0)

        # Canny Parameters
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

        # Prewitt Parameters
        self.prewittThreshold = QSpinBox()
        self.prewittThreshold.setRange(0, 255)
        self.prewittThreshold.setValue(50)

        self.prewittValue = QSpinBox()
        self.prewittValue.setRange(0, 255)
        self.prewittValue.setValue(255)

        # Apply Edge Detection Button
        self.btn_edge_detection = QPushButton("Detect Edges")
        self.btn_edge_detection.clicked.connect(self.detect_edges)

        # Layout for Edge Parameters
        self.edgeLayout = QHBoxLayout()
        self.edgeLayout.addWidget(self.edgeType)
        self.edgeLayout.addWidget(self.btn_edge_detection)
        self.layout.addLayout(self.edgeLayout)


                # Thresholding Section UI Components
        self.thresholdType = QComboBox()
        self.thresholdType.addItems(["global", "local"])
        # self.thresholdType.currentIndexChanged.connect(self.update_threshold_layout)

        # Global Thresholding
        self.globalThreshold = QSpinBox()
        self.globalThreshold.setRange(0, 255)
        self.globalThreshold.setValue(128)

        # Local Thresholding
        self.kernelSizeThreshold = QSpinBox()
        self.kernelSizeThreshold.setRange(1, 15)
        self.kernelSizeThreshold.setValue(4)

        self.kValue = QDoubleSpinBox()
        self.kValue.setRange(0.0, 5.0)
        self.kValue.setSingleStep(0.1)
        self.kValue.setValue(2.0)

        # Apply Thresholding Button
        self.btn_threshold = QPushButton("Apply Thresholding")
        self.btn_threshold.clicked.connect(self.apply_thresholding)

        # Layout for Thresholding Parameters
        self.thresholdLayout = QHBoxLayout()
        self.thresholdLayout.addWidget(self.thresholdType)
        self.thresholdLayout.addWidget(self.globalThreshold)  # Default (Global)
        self.thresholdLayout.addWidget(self.btn_threshold)
        self.layout.addLayout(self.thresholdLayout)
        
        # Histogram & Frequency Filtering Section
        self.btn_histogram = QPushButton("Show Histogram")
        self.btn_histogram.clicked.connect(self.show_histogram)
        self.layout.addWidget(self.btn_histogram)

        self.freqType = QComboBox()
        self.freqType.addItems(["low_pass", "high_pass"])
        self.btn_freq_filter = QPushButton("Apply Frequency Filter")
        self.btn_freq_filter.clicked.connect(self.apply_frequency_filter)

        freqLayout = QHBoxLayout()
        freqLayout.addWidget(self.freqType)
        freqLayout.addWidget(self.btn_freq_filter)
        self.layout.addLayout(freqLayout)

        # Hybrid Image Parameters
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

        # Apply Hybrid Image Button
        self.btn_hybrid = QPushButton("Create Hybrid Image")
        self.btn_hybrid.clicked.connect(self.create_hybrid_image)

        # Layout for Hybrid Image Parameters
        self.hybridLayout = QHBoxLayout()
        self.hybridLayout.addWidget(QLabel("Cutoff 1:"))
        self.hybridLayout.addWidget(self.cutoff1)
        self.hybridLayout.addWidget(QLabel("Type 1:"))
        self.hybridLayout.addWidget(self.type1)
        self.hybridLayout.addWidget(QLabel("Cutoff 2:"))
        self.hybridLayout.addWidget(self.cutoff2)
        self.hybridLayout.addWidget(QLabel("Type 2:"))
        self.hybridLayout.addWidget(self.type2)
        self.hybridLayout.addWidget(self.btn_hybrid)

        # Add Layout to Main Layout
        self.layout.addLayout(self.hybridLayout)

        # Equalization & Normalization Section
        self.btn_equalize = QPushButton("Equalize Image")
        self.btn_equalize.clicked.connect(self.equalize)
        self.layout.addWidget(self.btn_equalize)

        self.btn_normalize = QPushButton("Normalize Image")
        self.btn_normalize.clicked.connect(self.normalize)
        self.layout.addWidget(self.btn_normalize)

        # Active Contour (Snake)
        self.btn_snake = QPushButton("Active Contour (Snake)")
        # self.btn_snake.clicked.connect(self.run_snake)
        self.layout.addWidget(self.btn_snake)

        # Image Processing Control Buttons
        self.btn_confirm = QPushButton("Confirm Edit")
        self.btn_confirm.clicked.connect(self.confirm_edit)
        self.btn_discard = QPushButton("Discard Edit")
        self.btn_discard.clicked.connect(self.discard_edit)
        self.btn_reset = QPushButton("Reset Image")
        self.btn_reset.clicked.connect(self.reset_image)

        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.btn_confirm)
        controlLayout.addWidget(self.btn_discard)
        controlLayout.addWidget(self.btn_reset)
        self.layout.addLayout(controlLayout)

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
        noise_type = self.noiseType.currentText()
        self.modified_image = self.processors['noise'].add_noise(noise_type)
        self.display_image(self.modified_image)

    def apply_filter(self):
        filter_type = self.filterType.currentText()
        kernel_size = self.kernelSize.value()
        sigma = self.sigmaValue.value()
        self.modified_image = self.processors['noise'].apply_filters(filter_type, kernel_size=kernel_size, sigma=sigma)
        self.display_image(self.modified_image)

    def detect_edges(self):
        edge_type = self.edgeType.currentText()
        self.modified_image = self.processors['edge_detector'].detect_edges(edge_type)
        self.display_image(self.modified_image)

    def apply_thresholding(self):
        threshold_type = self.thresholdType.currentText()
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
        if hybird == True:
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
            edge_map = self.processors['edge_detector'].detect_edges(edge_type, **kwargs) # sobel (kernal_size=3, sigma=1.0), canny (low_threshold=50, high_threshold=150, max_edge_val=255, min_edge_val=0), prewitt (threshold=50, value=255), roberts
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
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

