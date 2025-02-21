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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Image Processing App")

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)

        # Buttons
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        self.layout.addWidget(self.btn_load_image)

        self.btn_edge_detection = QPushButton("Edge Detection (Sobel)")
        self.btn_edge_detection.clicked.connect(self.detect_edges_sobel)
        self.layout.addWidget(self.btn_edge_detection)

        self.btn_canny = QPushButton("Edge Detection (Canny)")
        self.btn_canny.clicked.connect(self.detect_edges_canny)
        self.layout.addWidget(self.btn_canny)

        self.btn_noise = QPushButton("Add Noise & Filter")
        self.btn_noise.clicked.connect(self.noise_and_filter)
        self.layout.addWidget(self.btn_noise)

        self.btn_snake = QPushButton("Active Contour (Snake)")
        self.btn_snake.clicked.connect(self.run_snake)
        self.layout.addWidget(self.btn_snake)

        # Image display label
        self.lbl_image = QLabel("No Image Loaded")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_image)

        # Processors
        self.original_image = None  # Will hold the original image
        self.image = None  # Will hold a NumPy array for the loaded image
        self.modified_image = None  # Will hold the processed image
        self.extra_image = None  # Will hold the hybrid image
        #if you want to add new processor or modify one edit this list
        self.processors = {key: None for key in ['noise', 'edge_detector', 'thresholding', 'frequency', 'histogram', 'image']}
        for processor in self.processors.keys():
            self.processors[processor] = ProcessorFactory.create_processor(processor)

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
