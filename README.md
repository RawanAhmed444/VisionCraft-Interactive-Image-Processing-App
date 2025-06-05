# Real-time Image Studio

This application is a user-friendly platform for digital image processing, built with both C++/Qt and Python/PyQt. It provides an interactive graphical interface where users can load images and experiment with a wide variety of image processing techniques. All core algorithms—including noise addition, spatial and frequency domain filtering, edge detection, thresholding, histogram analysis, hybrid image creation, geometric shape detection (lines, circles, ellipses via Hough Transform), and active contour (snake) segmentation—are implemented entirely from scratch, without relying on external libraries like OpenCV for the core processing in Python. All operations are accessible through intuitive controls, making the application suitable for both educational and research purposes in computer vision and image analysis.

---

## Features

- **Noise Addition**
  - Add and visualize uniform, Gaussian, and salt & pepper noise to images

- **Image Filtering**
  - Apply spatial filters: average, Gaussian, and median 
  - Frequency domain filtering: low-pass and high-pass filters (DFT-based, implemented from scratch)

- **Edge Detection**
  - Detect edges using Sobel, Canny, Prewitt, and Roberts operators

- **Thresholding**
  - Perform global and local (adaptive) thresholding
  
- **Histogram Analysis**
  - Display, equalize, and normalize image histograms

- **Hybrid Images**
  - Create hybrid images by combining frequency components of two images

- **Hough Transform**
  - Detect lines, circles, and ellipses in images using the Hough Transform 

- **Active Contour (Snake)**
  - Segment objects using an interactive active contour (snake) algorithm

- **Interactive GUI**
  - User-friendly PyQt interface for loading, processing, and saving images, with real-time visualization of results

_All core image processing algorithms are implemented from scratch, without using OpenCV for the main processing steps in Python._

---

## Project Structure

```
Qt/      # C++/Qt application
pyQt/    # Python/PyQt application
  ├── data/
  ├── docs/
  ├── notebooks/
  ├── resources/
  ├── src/
  ├── tests/
  ├── ui/
```

---

## Screenshots

<!-- Add screenshots or demo images here -->
*Main Window*

![image](https://github.com/user-attachments/assets/292c209f-e657-4f6b-8697-15288b7a5dc5)

*Canny Edge Detection*

![image](https://github.com/user-attachments/assets/6cfdce7e-aa6d-4525-93fb-f17f250ebb02)

*Line Hough Transform*

![image](https://github.com/user-attachments/assets/7c25c12e-8905-4518-9cef-60d7ab361e80)

*Circle Hough Transform*

![image](https://github.com/user-attachments/assets/62ba3278-3e17-4ce5-bb1f-57a49605b996)

*Ellipse Detection*

![image](https://github.com/user-attachments/assets/11ec6655-0d3f-4896-87a5-ec520dd5b215)

*Active Contour*

![image](https://github.com/user-attachments/assets/cb544736-c29d-419b-b9ad-01f0d3f4c67b)

---

## Getting Started

### Prerequisites

- For **Qt/C++**: Qt 6.x, OpenCV, CMake
- For **PyQt/Python**: Python 3.8+, PyQt5, OpenCV, NumPy, Matplotlib

### Installation

#### Python/PyQt

```sh
cd pyQt
pip install -r requirements.txt
python src/main.py
```

#### C++/Qt

```sh
cd Qt
mkdir build && cd build
cmake ..
make
./NoisyVisionsApp
```

---

## Contributors

* **Rawan Shoaib**: [GitHub Profile](https://github.com/RawanAhmed444)
* **Ahmed Aldeeb**: [GitHub Profile](https://github.com/AhmedXAlDeeb)
* **Ahmed Mahmoud**: [GitHub Profile](https://github.com/ahmed-226)
* **Eman Emad**: [GitHub Profile](https://github.com/Alyaaa16)

---

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [PyQt Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [Qt Documentation](https://doc.qt.io/)

*You can also have a look at our [report](https://drive.google.com/file/d/1zXFt8HgaLSsFsPe_RzEfZugyCzlSpCHg/view?usp=sharing)*

