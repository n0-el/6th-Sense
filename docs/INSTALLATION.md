# 📦 Installation Guide

Complete installation instructions for all supported platforms.

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Installation](#quick-installation)
3. [Platform-Specific Setup](#platform-specific-setup)
   - [NVIDIA Jetson](#nvidia-jetson)
   - [Raspberry Pi](#raspberry-pi)
   - [Ubuntu Desktop](#ubuntu-desktop)
   - [Windows](#windows)
4. [Troubleshooting](#troubleshooting)

## Requirements

### Hardware Requirements

**Minimum:**
- Camera: 720p @ 30fps
- CPU: Quad-core ARM or x86
- RAM: 4GB
- Storage: 8GB free space
- Audio: Speaker or headphones

**Recommended:**
- NVIDIA Jetson Nano or Xavier
- Camera: 1080p @ 30fps
- RAM: 8GB+
- SSD storage

### Software Requirements

- Python 3.8 or higher
- OpenCV 4.5+
- PyTorch 1.10+
- CUDA 10.2+ (for GPU acceleration)

## Quick Installation

For Ubuntu/Debian-based systems with GPU support:

```bash
# Clone repository
git clone https://github.com/yourusername/blind_spot_detection_system.git
cd blind_spot_detection_system

# Run automated setup
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Test installation
cd src
python main.py --help
```

## Platform-Specific Setup

### NVIDIA Jetson

#### Jetson Nano

**1. Flash JetPack 4.6+**

Download from: https://developer.nvidia.com/embedded/jetpack

**2. Initial Setup**

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Set to max power mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Install dependencies
sudo apt-get install -y python3-pip libopencv-dev
```

**3. Install PyTorch**

```bash
# For JetPack 4.6
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

**4. Install TensorRT (included in JetPack)**

```bash
# Verify installation
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

**5. Run Setup Script**

```bash
./setup.sh
```

**6. Optimize for Jetson**

Edit `config/config.yaml`:
```yaml
detection:
  use_tensorrt: true
  use_fp16: true
  input_size: [640, 640]
```

#### Jetson Xavier

Same as Nano, but can handle larger models:

```yaml
detection:
  model_path: "models/yolov8s.pt"  # Can use 'small' instead of 'nano'
  input_size: [640, 640]
```

### Raspberry Pi

#### Raspberry Pi 4 (8GB Recommended)

**1. Install Raspberry Pi OS (64-bit)**

**2. Enable Camera**

```bash
sudo raspi-config
# Select: Interfacing Options → Camera → Enable
```

**3. Increase Swap**

```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**4. Install Dependencies**

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv libatlas-base-dev
```

**5. Install PyTorch (Lightweight)**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**6. Run Setup**

```bash
./setup.sh
```

**7. Optimize for Pi**

Edit `config/config.yaml`:
```yaml
detection:
  model_path: "models/yolov8n.pt"
  frame_skip: 3  # Process every 3rd frame
  input_size: [416, 416]  # Smaller input

stabilization:
  enabled: false  # Disable for speed

visualization:
  enabled: false  # Disable GUI for headless operation
```

**8. Use Lite Model (Optional)**

```bash
pip3 install ultralytics-lite
```

### Ubuntu Desktop

#### With NVIDIA GPU

**1. Install CUDA and cuDNN**

Follow: https://developer.nvidia.com/cuda-downloads

**2. Install PyTorch with CUDA**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Run Setup**

```bash
./setup.sh
```

**4. Verify GPU**

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

#### CPU Only

**1. Install PyTorch (CPU)**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**2. Run Setup**

```bash
./setup.sh
```

**3. Configure for CPU**

Edit `config/config.yaml`:
```yaml
system:
  device: "cpu"

detection:
  use_tensorrt: false
  frame_skip: 2
```

### Windows

**1. Install Python 3.8+**

Download from: https://www.python.org/downloads/

**2. Install Visual Studio Build Tools**

Required for some packages.

**3. Install PyTorch**

```powershell
# With CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio
```

**4. Install Dependencies**

```powershell
pip install -r requirements.txt
```

**5. Download Model**

```powershell
cd models
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**6. Run System**

```powershell
cd src
python main.py
```

## Troubleshooting

### "ImportError: No module named 'cv2'"

```bash
pip install opencv-python
```

### "RuntimeError: CUDA out of memory"

Reduce batch size or input resolution:
```yaml
detection:
  input_size: [416, 416]  # Smaller
```

### "Cannot open camera"

**Linux:**
```bash
# Check available cameras
ls /dev/video*

# Grant camera permissions
sudo usermod -a -G video $USER
# Logout and login again
```

**Check in Python:**
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
```

### Low FPS

1. **Enable frame skipping:**
```yaml
detection:
  frame_skip: 3
```

2. **Reduce input size:**
```yaml
detection:
  input_size: [416, 416]
```

3. **Disable stabilization:**
```yaml
stabilization:
  enabled: false
```

4. **Use TensorRT (Jetson only):**
```bash
cd scripts
python convert_to_tensorrt.py
```

### Audio Not Working

**Linux:**
```bash
# Test audio
speaker-test -t wav -c 2

# Install audio libraries
sudo apt-get install portaudio19-dev libsndfile1
pip install sounddevice soundfile --force-reinstall
```

**Check devices:**
```python
import sounddevice as sd
print(sd.query_devices())
```

### "Model file not found"

```bash
cd scripts
./download_model.sh n
```

## Verification

After installation, verify everything works:

```bash
cd tests
python test_modules.py
```

Expected output:
```
✓ All tests passed
✓ OpenCV: 4.5.0
✓ PyTorch: 1.10.0
✓ Model loaded
```

## Performance Tuning

See `docs/PERFORMANCE.md` for detailed optimization guides.

## Next Steps

1. Configure system: Edit `config/config.yaml`
2. Test with webcam: `python main.py --source 0`
3. Mount camera on helmet
4. Test in safe environment
5. Fine-tune thresholds

## Getting Help

- GitHub Issues: Report bugs
- Documentation: Check `docs/` folder
- Discord: Join community

## Updates

Keep system updated:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

**Installation complete! Ready to ride safely! 🏍️**
