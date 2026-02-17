#!/bin/bash

# Setup script for Blind Spot Detection System
# Supports: Ubuntu 20.04+, Jetson (JetPack 4.6+), Raspberry Pi OS, Docker

set -e

echo "=================================="
echo "Blind Spot Detection System Setup"
echo "=================================="
echo ""

# Detect Docker
if [ -f /.dockerenv ]; then
    PLATFORM="docker"
    echo "Detected: Docker Container"
elif [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
    echo "Detected: NVIDIA Jetson"
elif [ -f /proc/cpuinfo ] && grep -q "Raspberry Pi" /proc/cpuinfo; then
    PLATFORM="raspberry_pi"
    echo "Detected: Raspberry Pi"
else
    PLATFORM="generic"
    echo "Detected: Generic Linux"
fi

echo ""

# Check if sudo exists
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

# ---------------------------
# System Update & Base Install
# ---------------------------

echo "Updating system packages..."
$SUDO apt-get update
$SUDO apt-get upgrade -y

echo "Installing system dependencies..."
$SUDO apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libopencv-dev \
    libportaudio2 \
    libsndfile1 \
    git \
    wget \
    curl

# ---------------------------
# Platform-Specific Setup
# ---------------------------

if [ "$PLATFORM" == "jetson" ]; then
    echo "Installing Jetson-specific packages..."

    $SUDO apt-get install -y \
        libcudnn8 \
        libcublas-dev \
        libnvinfer8 \
        libnvinfer-plugin8

    # Hardware tuning (only on real Jetson, not Docker)
    if [ "$PLATFORM" != "docker" ]; then
        $SUDO nvpmodel -m 0 || true
        $SUDO jetson_clocks || true
    fi

    echo "Jetson optimizations applied!"

elif [ "$PLATFORM" == "raspberry_pi" ]; then
    echo "Installing Raspberry Pi optimizations..."

    # Only run hardware commands if not Docker
    if [ "$PLATFORM" != "docker" ]; then
        $SUDO raspi-config nonint do_camera 0 || true

        $SUDO dphys-swapfile swapoff || true
        $SUDO sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile || true
        $SUDO dphys-swapfile setup || true
        $SUDO dphys-swapfile swapon || true
    fi

    echo "Raspberry Pi optimizations applied!"
fi

# ---------------------------
# Python Environment
# ---------------------------

echo "Setting up Python environment..."
cd "$(dirname "$0")"

if [ "$PLATFORM" == "docker" ]; then
    echo "Docker detected - installing packages globally"
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# ---------------------------
# Download YOLOv8 Model
# ---------------------------

echo "Downloading YOLOv8 model..."
mkdir -p models
cd models

if [ ! -f yolov8n.pt ]; then
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    echo "Model downloaded successfully!"
else
    echo "Model already exists, skipping download."
fi

cd ..

# ---------------------------
# Create Output Directories
# ---------------------------

echo "Creating output directories..."
mkdir -p data/test_videos
mkdir -p data/outputs

# ---------------------------
# Test Installation
# ---------------------------

echo ""
echo "Testing installation..."

python3 - <<EOF
import cv2
import numpy as np
import torch
from ultralytics import YOLO

print("✓ OpenCV:", cv2.__version__)
print("✓ NumPy:", np.__version__)
print("✓ PyTorch:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
EOF

echo ""
echo "=================================="
echo "Setup completed successfully!"
echo "=================================="
echo ""

if [ "$PLATFORM" != "docker" ]; then
    echo "To activate the environment:"
    echo "  source venv/bin/activate"
fi

echo ""
echo "To run the system:"
echo "  cd src"
echo "  python main.py"
echo ""
echo "For help:"
echo "  python main.py --help"
echo ""