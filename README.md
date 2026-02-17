# 🏍️ Smart Motorcycle Helmet Blind Spot Detection System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)

A real-time AI-powered blind spot detection system for motorcycle helmets using computer vision and edge computing.

## 🎯 Features

- **Real-time Vehicle Detection** - YOLOv8-nano optimized for edge devices
- **Multi-Object Tracking** - ByteTrack algorithm for consistent tracking
- **Danger Estimation** - Time-to-Collision (TTC) calculation without radar
- **Spatial Audio Alerts** - Directional warnings (left/right/center)
- **Edge Optimized** - Runs on Jetson Nano, Raspberry Pi, or CPU
- **Robust to Motion** - Handles helmet vibration and movement
- **False Positive Suppression** - Filters traffic jams, parked cars, lane changes

## 📊 System Architecture

```
Camera Input (30 FPS)
    ↓
Video Stabilization (Optional)
    ↓
Vehicle Detection (YOLOv8-nano)
    ↓
Multi-Object Tracking (ByteTrack)
    ↓
Danger Estimation (TTC Calculation)
    ↓
Decision Engine (FSM + Smoothing)
    ↓
Audio Alert System (Spatial Audio)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Camera (USB webcam or CSI camera for Jetson)
- Audio output device
- (Optional) NVIDIA Jetson Nano/Xavier for GPU acceleration

### Installation

1. **Clone the repository**
```bash
cd blind_spot_detection_system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download YOLOv8 model**
```bash
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
cd ..
```

4. **Configure the system**

Edit `config/config.yaml` to match your hardware:

```yaml
camera:
  source: 0  # Camera index or video file path
  width: 1280
  height: 720
  fps: 30

detection:
  model_path: "models/yolov8n.pt"
  confidence_threshold: 0.4
```

### Running the System

**Basic usage (camera):**
```bash
cd src
python main.py
```

**Using video file:**
```bash
python main.py --source ../data/test_videos/sample.mp4
```

**Save output video:**
```bash
python main.py --source 0 --output ../data/outputs/output.mp4
```

**Custom config:**
```bash
python main.py --config custom_config.yaml
```

### Controls

- **Q** - Quit
- **R** - Reset system

## 📁 Project Structure

```
blind_spot_detection_system/
├── config/
│   └── config.yaml              # Main configuration file
├── src/
│   ├── modules/
│   │   ├── stabilization.py     # Video stabilization
│   │   ├── detection.py         # Vehicle detection (YOLOv8)
│   │   ├── tracking.py          # Multi-object tracking (ByteTrack)
│   │   ├── danger_estimation.py # TTC and danger classification
│   │   ├── decision.py          # Decision engine
│   │   └── audio.py             # Spatial audio alerts
│   ├── utils/
│   │   └── helpers.py           # Visualization and logging
│   └── main.py                  # Main system entry point
├── models/
│   └── yolov8n.pt               # YOLOv8-nano model (download)
├── data/
│   ├── test_videos/             # Test video files
│   └── outputs/                 # System outputs (videos, logs)
├── tests/
│   └── test_modules.py          # Unit tests
├── docs/
│   ├── INSTALLATION.md          # Detailed installation guide
│   ├── CONFIGURATION.md         # Configuration options
│   └── PERFORMANCE.md           # Performance optimization guide
├── scripts/
│   ├── download_model.sh        # Download YOLOv8 model
│   ├── setup_jetson.sh          # Jetson Nano setup
│   └── test_audio.py            # Test audio system
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## ⚙️ Configuration

### Key Configuration Options

**Detection Settings:**
```yaml
detection:
  confidence_threshold: 0.4    # Detection confidence (0.0-1.0)
  frame_skip: 2                # Process every Nth frame
  use_tensorrt: true           # Use TensorRT (Jetson only)
```

**Danger Thresholds:**
```yaml
danger_estimation:
  ttc_safe: 5.0       # Safe TTC threshold (seconds)
  ttc_warning: 3.0    # Warning TTC threshold
  ttc_critical: 1.5   # Critical TTC threshold
```

**Audio Alerts:**
```yaml
audio_alerts:
  enabled: true
  warning_freq: 1500  # Warning tone frequency (Hz)
  critical_freq: 2000 # Critical tone frequency (Hz)
```

See `docs/CONFIGURATION.md` for complete options.

## 🎬 Testing

### Test Individual Modules

**Test audio system:**
```bash
cd src/modules
python audio.py
```

**Test detection:**
```bash
cd src
python -c "from modules.detection import VehicleDetector; import yaml; config = yaml.safe_load(open('../config/config.yaml'))['detection']; detector = VehicleDetector(config)"
```

### Run Unit Tests

```bash
cd tests
pytest test_modules.py
```

### Benchmark Performance

```bash
cd scripts
python benchmark.py --device jetson  # or cpu, gpu
```

## 📈 Performance

### Target Metrics

| Platform | FPS | Latency | Power |
|----------|-----|---------|-------|
| Jetson Nano | 13-15 | 70-80ms | 10W |
| Jetson Xavier | 25-30 | 35-40ms | 15W |
| Raspberry Pi 4 | 8-10 | 100-120ms | 8W |
| CPU (i7) | 15-20 | 50-60ms | 45W |

### Optimization Tips

1. **Use TensorRT** (Jetson only):
```bash
python scripts/convert_to_tensorrt.py
```

2. **Reduce input resolution**:
```yaml
detection:
  input_size: [640, 384]  # Smaller = faster
```

3. **Increase frame skipping**:
```yaml
detection:
  frame_skip: 3  # Process every 3rd frame
```

4. **Disable stabilization** (if not needed):
```yaml
stabilization:
  enabled: false
```

See `docs/PERFORMANCE.md` for detailed optimization guide.

## 🔧 Hardware Setup

### Recommended Hardware

**For Jetson Nano:**
- NVIDIA Jetson Nano Developer Kit (4GB)
- USB Camera or CSI Camera (1280x720 @ 30fps)
- Bluetooth speaker or wired headphones
- MicroSD card (64GB+)
- Power supply (5V 4A)

**For Raspberry Pi:**
- Raspberry Pi 4 (8GB recommended)
- Pi Camera v2 or USB camera
- Audio output (3.5mm jack or USB)

### Camera Mounting

- Mount camera on **rear of helmet** facing backward
- Angle camera ~10° downward for better road coverage
- Secure with adhesive mount or helmet bracket
- Ensure lens is clean and unobstructed

### Power Supply

- Use portable USB power bank (10,000+ mAh)
- Mount in jacket pocket or tank bag
- Use right-angle USB cables to reduce bulk

## 🛠️ Development

### Adding Custom Models

1. Export your model to ONNX format
2. Place in `models/` directory
3. Update config:
```yaml
detection:
  model_path: "models/your_model.pt"
```

### Extending Functionality

**Add new alert types:**

Edit `src/modules/audio.py`:
```python
def generate_custom_alert(self, frequency, pattern):
    # Your alert logic here
    pass
```

**Modify danger thresholds:**

Edit `src/modules/danger_estimation.py`:
```python
def classify_danger(self, track):
    # Custom danger classification
    pass
```

## 📝 Logging and Analysis

### Event Logs

System saves JSON event logs to `data/outputs/events.json`:

```json
{
  "timestamp": 1234567890.123,
  "type": "alert",
  "level": "CRITICAL",
  "track_id": 5,
  "class": "car",
  "ttc": 1.2,
  "position": "LEFT"
}
```

### Performance Metrics

CSV metrics saved to `data/outputs/metrics.csv`:

```
timestamp,frame,detection_ms,tracking_ms,total_latency_ms,fps
1234567890.1,1,55.2,4.1,73.5,13.7
```

### Analysis Scripts

```bash
cd scripts
python analyze_logs.py --events ../data/outputs/events.json
```

## 🚨 Safety Notice

**⚠️ IMPORTANT SAFETY INFORMATION:**

- This system is a **supplemental aid** - DO NOT rely on it exclusively
- Always check blind spots manually before lane changes
- System may have false positives and false negatives
- Weather conditions (rain, fog) may affect performance
- Rider alertness and proper riding technique are paramount
- Test thoroughly in safe environments before road use
- Comply with all local traffic laws and regulations

## 🔬 Research and Development

### Dataset Collection

Collect your own dataset:
```bash
cd scripts
python collect_data.py --camera 0 --output ../data/collected/
```

### Model Training

Fine-tune on your data:
```bash
cd scripts
python train_custom_model.py --data ../data/collected/ --epochs 100
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See `CONTRIBUTING.md` for guidelines.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- ByteTrack algorithm
- OpenCV community
- NVIDIA Jetson platform

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Join our Discord server
- **Email**: support@example.com

## 🗺️ Roadmap

### Version 1.1 (Q2 2024)
- [ ] V2V communication support
- [ ] Radar fusion
- [ ] Haptic feedback integration
- [ ] Mobile app for monitoring

### Version 2.0 (Q4 2024)
- [ ] Night vision (thermal camera)
- [ ] Multi-camera support
- [ ] Cloud-based fleet analytics
- [ ] Lane-keeping assistance

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Configuration Reference](docs/CONFIGURATION.md)
- [Performance Tuning](docs/PERFORMANCE.md)
- [API Documentation](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 📊 Benchmarks

See `docs/BENCHMARKS.md` for detailed performance comparisons across different hardware platforms.

---

**Built with ❤️ for safer riding**

🏍️ Ride safe, stay aware! 🏍️
