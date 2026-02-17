import cv2
import numpy as np
import time
import os
import shutil
import sys

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from modules.event_recorder import EventRecorder

def test_event_recorder():
    print("Testing EventRecorder...")
    
    # Setup
    output_dir = "tests/test_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    config = {
        'enabled': True,
        'record_fps': 10,
        'pre_seconds': 2,
        'post_seconds': 2,
        'output_dir': output_dir,
        'save_width': 640,
        'save_height': 360
    }
    
    recorder = EventRecorder(config)
    
    # Create dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 1. Feed pre-event frames
    print("Feeding pre-event frames...")
    for i in range(20): # 2 seconds * 10 fps
        # Draw frame number
        f = frame.copy()
        cv2.putText(f, f"Pre {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        recorder.add_frame(f)
        time.sleep(0.01)
        
    # 2. Trigger Alert
    print("Triggering alert...")
    alerts = [{
        'level': 'CRITICAL',
        'position': 'CENTER',
        'ttc': 1.5,
        'class_name': 'car'
    }]
    recorder.on_alert(alerts, frame)
    
    # 3. Feed post-event frames
    print("Feeding post-event frames...")
    for i in range(30): # 3 seconds (more than post_seconds to ensure completion)
        f = frame.copy()
        cv2.putText(f, f"Post {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        recorder.add_frame(f)
        time.sleep(0.01)

    # 4. Stop
    print("Stopping recorder...")
    recorder.stop()
    
    # 5. Verify
    files = os.listdir(output_dir)
    mp4_files = [f for f in files if f.endswith('.mp4')]
    
    if len(mp4_files) == 1:
        print(f"SUCCESS: Created file {mp4_files[0]}")
        file_path = os.path.join(output_dir, mp4_files[0])
        size = os.path.getsize(file_path)
        print(f"File size: {size} bytes")
        if size > 1000:
            return True
        else:
            print("FAILURE: File too small")
            return False
    else:
        print(f"FAILURE: Expected 1 mp4 file, found {len(mp4_files)}")
        return False

if __name__ == "__main__":
    try:
        if test_event_recorder():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"EXCEPTION: {e}")
        sys.exit(1)
