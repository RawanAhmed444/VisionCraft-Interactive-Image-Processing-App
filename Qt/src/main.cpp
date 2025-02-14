#include <QApplication>
#include <QLabel>
#include <QPixmap>
#include "mainwindow.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // OpenCV Test
    cv::Mat image = cv::imread("../data/test_image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Convert OpenCV Mat to Qt QImage
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    QImage qImage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    
    // Display image using QLabel
    QLabel imageLabel;
    imageLabel.setPixmap(QPixmap::fromImage(qImage));
    imageLabel.setWindowTitle("Qt + OpenCV Image Display");
    imageLabel.show();

    // Qt Test
    MainWindow window;
    window.setWindowTitle("Qt Main Window Test");
    window.show();

    return app.exec();
}
