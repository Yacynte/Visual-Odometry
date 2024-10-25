# Visual-Odometry (Work in Progress)
Stereo Visual Odometry Algorithm A Python-based stereo visual odometry algorithm that estimates camera motion by processing consecutive stereo image frames. The algorithm leverages depth triangulation and feature extraction to provide accurate pose estimates in meters, using the KITTI dataset for testing and validation.

# Stereo Visual Odometry Algorithm

## Overview
This repository contains a stereo visual odometry (VO) algorithm designed to estimate camera motion by processing consecutive frames from a stereo camera setup. The algorithm calculates both rotation and translation, using the KITTI dataset for testing and validation, providing results in meters for real-world applicability.

## Features
- **Stereo Image Processing**: Utilizes left and right camera images for depth triangulation.
- **Motion Estimation**: Provides accurate rotation (R) and translation (T) estimates between frames.
- **KITTI Dataset Compatibility**: Optimized to work with the `2011_09_26_drive_0013_extract` sequence from the KITTI dataset, including use of camera parameters and distortion coefficients.
- **Translation in Meters**: Outputs camera motion in real-world metric units.

## Prerequisites
- **Python** >= 3.x
- **OpenCV**: For feature detection and image processing.
- **Open3D**: For 3D point cloud visualization.
- **NumPy**: For efficient matrix operations.
- **KITTI Dataset**: Download and use `2011_09_26_drive_0013_extract`.

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stereo-visual-odometry.git
   cd stereo-visual-odometry
2. Install the dependencies
    ```bash
    pip install -r requirements.txt

### Usage
1. Place your KITTI dataset files in the Visual-Odometry folder.
2.  Run the stereo visual odometry algorithm
    ```bash
    python visual_odometry.py

### Algorithm Details
The algorithm is divided into two main components:

1. Stereo Visual Odometry: Processes image frames from the stereo cameras, triangulates depth, and estimates camera pose relative to the previous frame.
2. Rotation and Translation Calculation: Calculates R and T transformations from 2D feature correspondences across consecutive frames.

## Results
The algorithm provides the following outputs:

1. Camera Path Reconstruction: A visual path reconstruction representing the motion of the camera in 3D space.
2. Translation Measurements: Outputs translation in meters, which can be used for mapping or distance estimation in real-world applications.
Sample results can be visualized as a 3D path of the camera's motion, with each frame's pose represented as a point in 3D space. The calculated R and T values between frames provide insights into how the camera moved, which can be applied in tasks like autonomous navigation, environmental mapping, and 3D reconstruction.

### Future Improvements
1. Improved depth accuracy with advanced feature matching techniques.
2. Real-time integration and testing on live stereo camera systems.