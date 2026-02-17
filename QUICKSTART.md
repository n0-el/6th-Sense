# 🚀 Quick Start Guide

Get the Blind Spot Detection System running in 5 minutes!

## Step 1: Install Dependencies

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

This will:
- Install system packages
- Create Python virtual environment
- Install Python dependencies
- Download YOLOv8 model
- Create necessary directories

## Step 2: Activate Environment

```bash
source venv/bin/activate
```

## Step 3: Test with Webcam

```bash
cd src
python main.py --source 0
```

You should see:
- Live camera feed
- Bounding boxes around detected vehicles
- Danger level indicators (Green/Yellow/Red)
- FPS and latency metrics

## Step 4: Test with Video File

Don't have a helmet camera yet? Test with a video:

```bash
# Download a sample dashcam video
cd ../data/test_videos
youtube-dl "YOUR_DASHCAM_VIDEO_URL" -o sample.mp4

# Or use your own video file
cp /path/to/your/video.mp4 sample.mp4

# Run system
cd ../../src
python main.py --source ../data/test_videos/sample.mp4
```

## Step 5: Test Audio Alerts

```bash
cd src/modules
python audio.py
```

You should hear:
- LEFT warning beep
- RIGHT critical beep
- CENTER critical beep

## Controls

While system is running:

- **Q** - Quit the system
- **R** - Reset tracking

## Common Issues

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "Cannot open camera"
```bash
# Check available cameras
ls /dev/video*

# Try different camera index
python main.py --source 1
```

### "Model file not found"
```bash
cd scripts
./download_model.sh n
```

### Low FPS on Raspberry Pi
Edit `config/config.yaml`:
```yaml
detection:
  frame_skip: 3        # Process every 3rd frame
  input_size: [320, 320]  # Smaller input
```

## Next Steps

1. **Configure for your hardware**
   - Edit `config/config.yaml`
   - Adjust thresholds
   - Enable/disable features

2. **Mount camera on helmet**
   - Position facing rear
   - Secure mounting
   - Clean lens

3. **Test in controlled environment**
   - Parking lot test
   - Controlled approaches
   - Verify alerts

4. **Fine-tune settings**
   - Adjust TTC thresholds
   - Configure audio alerts
   - Optimize performance

## Performance Expectations

| Platform | FPS | Setup Time |
|----------|-----|------------|
| Jetson Nano | 13-15 | ~20 min |
| Raspberry Pi 4 | 8-10 | ~30 min |
| Desktop CPU | 15-20 | ~10 min |
| Desktop GPU | 30-40 | ~10 min |

## Getting Help

- Check `README.md` for full documentation
- See `docs/TROUBLESHOOTING.md` for solutions
- Open an issue on GitHub

## Safety Reminder

⚠️ **This is a supplemental aid only. Always check blind spots manually!**

Happy riding! 🏍️
