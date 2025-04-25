#!/bin/bash

# Update package lists
sudo apt-get update

# Install system dependencies for OpenCV and Python
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    python3-opencv

# Ensure pip is up to date
pip install --upgrade pip

# Install or upgrade required Python packages
pip install \
    opencv-python \
    numpy \
    ultralytics \
    torch \
    torchvision

# Verify installations
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
python3 -c "import ultralytics; print('Ultralytics version:', ultralytics.__version__)"