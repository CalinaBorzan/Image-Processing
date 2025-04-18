# Image Processing Application (C++ / OpenCV)

This is a C++ console application developed using OpenCV for basic and advanced image processing tasks. The project offers a menu-based interface to run various image operations, from grayscale conversions and edge detection to morphological operations, region filling, histogram equalization, object detection, and feature extraction.

## 🔧 Features

- 📂 Open individual images or folders
- 🎨 Convert images to grayscale, HSV, or negative
- ✂️ Resize images (with/without interpolation)
- ⚫ Canny edge detection and Gaussian blur
- 📸 Snap from webcam and process video sequences
- 🖱️ Mouse click for pixel value inspection
- ➕ Brightness adjustment (additive & multiplicative)
- 🟥 Region analysis:
  - Center of mass
  - Aspect ratio
  - Perimeter & thinness
  - Orientation angle
  - Horizontal & vertical projections
- 🧠 Binary segmentation:
  - Thresholding (manual & automatic)
  - Connected-component labeling (BFS, two-pass)
  - Border tracing (and chain code)
- ➕ Morphological operations:
  - Dilation
  - Erosion
  - Opening & closing (single & N-times)
  - Boundary extraction
  - Region filling
- 📊 Histogram:
  - Plotting
  - Equalization
  - Stretching/shrinking
  - Gamma correction
  - Mean & standard deviation



## 🚀 Getting Started

### Prerequisites

- Visual Studio 2022
- OpenCV 4.9.0 (or compatible version)
- CMake (optional)
- C++17 support

### How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/CalinaBorzan/Image-Processing-Clean.git
   ```
2. Open the `.sln` file in Visual Studio.
3. Build and run the project.
4. Use the menu in the console to explore the features.

## 🧠 Useful Info

- All image processing operations are handled via the OpenCV library.
- Menu navigation is keyboard-based and allows real-time switching between features.
- Some operations require image files to be present in the `Images/` or `Videos/` folders.
- This project was developed as part of an academic assignment for an image processing course.
