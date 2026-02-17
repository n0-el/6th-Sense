#!/bin/bash

# Script to download YOLOv8 models
# Usage: ./download_model.sh [model_size]
# model_size: n (nano), s (small), m (medium), l (large), x (xlarge)

MODEL_SIZE=${1:-n}  # Default to nano

echo "Downloading YOLOv8-${MODEL_SIZE}..."

cd "$(dirname "$0")/../models"

# YOLOv8 model URLs
BASE_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0"

case $MODEL_SIZE in
    n)
        MODEL_FILE="yolov8n.pt"
        ;;
    s)
        MODEL_FILE="yolov8s.pt"
        ;;
    m)
        MODEL_FILE="yolov8m.pt"
        ;;
    l)
        MODEL_FILE="yolov8l.pt"
        ;;
    x)
        MODEL_FILE="yolov8x.pt"
        ;;
    *)
        echo "Invalid model size: $MODEL_SIZE"
        echo "Valid options: n, s, m, l, x"
        exit 1
        ;;
esac

# Download
if [ -f "$MODEL_FILE" ]; then
    echo "Model $MODEL_FILE already exists. Skipping download."
else
    wget "$BASE_URL/$MODEL_FILE"
    echo "Downloaded $MODEL_FILE successfully!"
fi

# Update config
echo ""
echo "Update your config.yaml to use this model:"
echo "detection:"
echo "  model_path: \"models/$MODEL_FILE\""
