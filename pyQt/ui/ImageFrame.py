from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QLabel
from PyQt5.QtCore import Qt


class ImageFrame(QWidget):
    def __init__(self, title, radio_button_group, radio_button_checked=False):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        self.radio_button = QRadioButton("")
        self.radio_button.setStyleSheet("color: white; font: 600 12pt 'Segoe UI';border:none;")
        self.radio_button.setMaximumWidth(50)
        self.radio_button.setChecked(radio_button_checked)
        radio_button_group.addButton(self.radio_button)
        
        self.label = QLabel(title)
        self.label.setStyleSheet("background-color: black;font: 600 15pt 'Segoe UI';border:none;")
        self.label.setMaximumHeight(70)
        self.label.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel("")
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;border: none; font: 600 15pt 'Segoe UI';")
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.radio_button)
        h_layout.addWidget(self.label)
        
        self.layout.addLayout(h_layout)
        self.layout.addWidget(self.image_label)