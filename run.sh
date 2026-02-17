#!/bin/bash

# Smart run script for Blind Spot Detection System

set -e

echo "🏍️  Starting Blind Spot Detection System..."
echo ""

# Detect Docker
if [ -f /.dockerenv ]; then
    echo "Detected: Docker environment"
    PYTHON=python3
else
    echo "Detected: Local environment"

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Virtual environment not found. Running setup..."
        ./setup.sh
    fi

    # Activate virtual environment
    source venv/bin/activate
    PYTHON=python
fi

# Ensure model exists
if [ ! -f "models/yolov8n.pt" ]; then
    echo "Model not found. Downloading..."
    mkdir -p models
    cd models
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    cd ..
fi

# Run the system
cd src
$PYTHON main.py "$@"