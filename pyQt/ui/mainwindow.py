import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QTabWidget, QLabel, QPushButton, QRadioButton, QFileDialog, QSpacerItem, QSizePolicy, QButtonGroup
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 1800, 800)
        self.setStyleSheet("background-color: rgb(4, 3, 26); color: white;")

        # Main layout
        main_layout = QHBoxLayout()

        # Left frame with tab widget
        left_frame = QFrame()
        left_frame.setMinimumSize(300, 600)
        left_frame.setMaximumWidth(500)
        left_frame.setStyleSheet("background-color: rgb(4, 3, 26); color: white;")
        left_layout = QVBoxLayout(left_frame)

        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane { /* The tab widget frame */
                border: none;
                background-color: rgb(4, 3, 26);
            }
            QTabBar::tab {
                background:  rgb(4, 3, 26);
                color: white;
                padding: 10px;
                border: 1px solid rgb(4, 3, 26);
                border-radius: 5px;
                font: 600 10pt 'Segoe UI';
                width: 70px;
                height: 25px;
            }
            QTabBar::tab:selected {
                background: rgb(40, 57, 153);
                color: white;
            }
        """)

        filter_tab = QWidget()
        filter_tab.setMinimumHeight(800)
        filter_layout = QVBoxLayout(filter_tab)
        filter_layout.setSpacing(20)

        # Noise frame
        noise_frame = QFrame()
        noise_frame.setMinimumHeight(300)
        noise_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        noise_frame.setStyleSheet("border: 2px solid rgb(40, 57, 153); border-radius: 10px;padding: 10px;")
        noise_layout = QVBoxLayout(noise_frame)
        noise_layout.setAlignment(Qt.AlignTop)
        noise_layout.setSpacing(20)

        noise_label = QLabel("Noise")
        noise_label.setStyleSheet("border: none; font: 600 15pt 'Segoe UI';")
        noise_label.setAlignment(Qt.AlignCenter)
        noise_label.setMaximumHeight(70)
        noise_layout.addWidget(noise_label)

        gaussian_noise_button = QPushButton("Gaussian Noise")
        gaussian_noise_button.setMinimumHeight(50)
        gaussian_noise_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        uniform_noise_button = QPushButton("Uniform Noise")
        uniform_noise_button.setMinimumHeight(50)
        uniform_noise_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        salt_pepper_noise_button = QPushButton("Salt & Pepper Noise")
        salt_pepper_noise_button.setMinimumHeight(50)
        salt_pepper_noise_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")

        noise_layout.addWidget(gaussian_noise_button)
        noise_layout.addWidget(uniform_noise_button)
        noise_layout.addWidget(salt_pepper_noise_button)

        filters_frame = QFrame()
        filters_frame.setMinimumHeight(300)
        filters_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        filters_frame.setStyleSheet("border: 2px solid rgb(40, 57, 153); border-radius: 10px; padding: 10px;")
        filters_layout = QVBoxLayout(filters_frame)
        filters_layout.setAlignment(Qt.AlignTop)
        filters_layout.setSpacing(20)

        filters_label = QLabel("Filters")
        filters_label.setStyleSheet("border: none; font: 600 15pt 'Segoe UI';")
        filters_label.setAlignment(Qt.AlignCenter)
        filters_label.setMaximumHeight(70)
        filters_layout.addWidget(filters_label)

        radio_frame = QFrame()
        radio_frame.setStyleSheet("border: none;margin: 0px; padding: 0px;")
        radio_layout = QHBoxLayout(radio_frame)
        radio_layout.setAlignment(Qt.AlignCenter)
        radio_layout.setSpacing(20)

        radio_button_group = QButtonGroup()
        radio_3x3 = QRadioButton("3x3")
        radio_3x3.setStyleSheet("color: white; font: 600 12pt 'Segoe UI';")
        radio_5x5 = QRadioButton("5x5")
        radio_5x5.setStyleSheet("color: white; font: 600 12pt 'Segoe UI';")
        radio_button_group.addButton(radio_3x3)
        radio_button_group.addButton(radio_5x5)

        radio_layout.addWidget(radio_3x3)
        radio_layout.addWidget(radio_5x5)

        filters_layout.addWidget(radio_frame)

        gaussian_filter_button = QPushButton("Gaussian Filter")
        gaussian_filter_button.setMinimumHeight(50)
        gaussian_filter_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        median_filter_button = QPushButton("Median Filter")
        median_filter_button.setMinimumHeight(50)
        median_filter_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        average_filter_button = QPushButton("Average Filter")
        average_filter_button.setMinimumHeight(50)
        average_filter_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")

        filters_layout.addWidget(gaussian_filter_button)
        filters_layout.addWidget(median_filter_button)
        filters_layout.addWidget(average_filter_button)

        filter_layout.addWidget(noise_frame)
        filter_layout.addWidget(filters_frame)
        filter_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        histogram_tab = QWidget()
        histogram_layout = QVBoxLayout(histogram_tab)
        histogram_label = QLabel("Histogram Page")
        histogram_layout.addWidget(histogram_label)

        tab_widget.addTab(filter_tab, "Filter")
        tab_widget.addTab(histogram_tab, "Histogram")

        left_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))
        left_layout.addWidget(tab_widget)
        main_layout.addWidget(left_frame)

        right_frame = QFrame()
        right_frame.setMinimumSize(800, 600)
        right_frame.setStyleSheet("background-color: rgb(4, 3, 26); color: white;")
        right_layout = QVBoxLayout(right_frame)

        input_output_frame = QFrame()
        input_output_layout = QHBoxLayout(input_output_frame)
        input_output_layout.setSpacing(20)


        self.input_frame = QFrame()
        self.input_frame.setStyleSheet("background-color: black; border: 2px solid rgb(40, 57, 153); border-radius: 10px; padding: 10px;")
        self.input_layout = QVBoxLayout(self.input_frame)

        self.input_label = QLabel("Input Image")
        self.input_label.setStyleSheet("background-color: black;font: 600 15pt 'Segoe UI';border:none;")
        self.input_label.setMaximumHeight(70)
        self.input_label.setAlignment(Qt.AlignCenter)

        self.input_image_label = QLabel("")
        self.input_image_label.setAlignment(Qt.AlignTop)
        self.input_image_label.setStyleSheet("background-color: black;border: none; font: 600 15pt 'Segoe UI';")

        self.input_layout.addWidget(self.input_label)
        self.input_layout.addWidget(self.input_image_label)
        
        input_output_layout.addWidget(self.input_frame)


        self.output_frame = QFrame()
        self.output_frame.setStyleSheet("background-color: black; border: 2px solid rgb(40, 57, 153); border-radius: 10px; padding: 10px;")
        self.output_layout = QVBoxLayout(self.output_frame)

        self.output_label = QLabel("Output Image")
        self.output_label.setStyleSheet("background-color: black;font: 600 15pt 'Segoe UI';border:none;")
        self.output_label.setMaximumHeight(70)
        self.output_label.setAlignment(Qt.AlignCenter)

        self.output_image_label = QLabel("")
        self.output_image_label.setAlignment(Qt.AlignTop)
        self.output_image_label.setStyleSheet("background-color: black;border: none; font: 600 15pt 'Segoe UI';")

        self.output_layout.addWidget(self.output_label)
        self.output_layout.addWidget(self.output_image_label)

        input_output_layout.addWidget(self.output_frame)



        control_frame = QFrame()
        control_frame.setMaximumHeight(60)
        control_layout = QHBoxLayout(control_frame)
        control_layout.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        spacer = QSpacerItem(50, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        control_layout.addItem(spacer)

        load_button = QPushButton("Load Image")
        load_button.setMinimumSize(100, 50)
        load_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        load_button.clicked.connect(self.load_image)

        clear_button = QPushButton("Clear Image")
        clear_button.setMinimumSize(100, 50)
        clear_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        clear_button.clicked.connect(self.clear_image)

        control_layout.addWidget(load_button)
        control_layout.addWidget(clear_button)

        right_layout.addWidget(control_frame)
        right_layout.addWidget(input_output_frame)
        main_layout.addWidget(right_frame)

        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.input_image_label.setPixmap(pixmap.scaled(self.input_image_label.width(), self.input_image_label.height(), Qt.AspectRatioMode.KeepAspectRatio))
            self.output_image_label.setPixmap(pixmap.scaled(self.output_image_label.width(), self.output_image_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def clear_image(self):
        self.input_image_label.clear()
        self.output_label.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())