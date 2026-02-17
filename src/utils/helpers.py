"""
Utility Functions
Helper functions for visualization, logging, and performance monitoring
"""

import cv2
import numpy as np
import time
from collections import deque
from typing import List, Dict, Tuple
import json
import csv
from pathlib import Path


class FPSTracker:
    """Track and display FPS"""
    
    def __init__(self, window_size: int = 30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        """Update with new frame"""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def get_fps(self) -> float:
        """Get average FPS"""
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_time = np.mean(self.frame_times)
        if avg_time == 0:
            return 0.0
        
        return 1.0 / avg_time


class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, log_file: str = None):
        self.metrics = {
            'frame_count': 0,
            'detection_time': deque(maxlen=100),
            'tracking_time': deque(maxlen=100),
            'total_latency': deque(maxlen=100),
            'fps': deque(maxlen=100)
        }
        
        self.log_file = log_file
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'frame', 'detection_ms', 'tracking_ms',
                    'total_latency_ms', 'fps'
                ])
    
    def update(self, detection_time: float, tracking_time: float,
               total_latency: float, fps: float):
        """Update metrics"""
        self.metrics['frame_count'] += 1
        self.metrics['detection_time'].append(detection_time)
        self.metrics['tracking_time'].append(tracking_time)
        self.metrics['total_latency'].append(total_latency)
        self.metrics['fps'].append(fps)
        
        # Log to file
        if self.log_file:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    self.metrics['frame_count'],
                    detection_time,
                    tracking_time,
                    total_latency,
                    fps
                ])
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'frames': self.metrics['frame_count'],
            'avg_detection_ms': np.mean(self.metrics['detection_time']) if self.metrics['detection_time'] else 0,
            'avg_tracking_ms': np.mean(self.metrics['tracking_time']) if self.metrics['tracking_time'] else 0,
            'avg_latency_ms': np.mean(self.metrics['total_latency']) if self.metrics['total_latency'] else 0,
            'avg_fps': np.mean(self.metrics['fps']) if self.metrics['fps'] else 0
        }


class EventLogger:
    """Log detection and alert events"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.events = []
    
    def log_alert(self, alert: Dict):
        """Log an alert event"""
        event = {
            'timestamp': time.time(),
            'type': 'alert',
            'level': alert['level'],
            'track_id': alert['track_id'],
            'class': alert.get('class_name', 'unknown'),
            'ttc': alert.get('ttc', None),
            'position': alert.get('position', None)
        }
        
        self.events.append(event)
    
    def log_detection(self, num_detections: int, num_tracks: int):
        """Log detection statistics"""
        event = {
            'timestamp': time.time(),
            'type': 'detection',
            'num_detections': num_detections,
            'num_tracks': num_tracks
        }
        
        self.events.append(event)
    
    def save(self):
        """Save events to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.events, f, indent=2)


class Visualizer:
    """Visualization utilities"""
    
    # Colors (BGR)
    COLOR_SAFE = (0, 255, 0)
    COLOR_WARNING = (0, 255, 255)
    COLOR_CRITICAL = (0, 0, 255)
    COLOR_ROI = (255, 0, 255)
    COLOR_TEXT = (255, 255, 255)
    
    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: List, danger_levels: List[str],
                   estimator=None) -> np.ndarray:
        """
        Draw tracks with danger level coloring.
        
        Args:
            frame: Input frame
            tracks: List of Track objects
            danger_levels: Danger levels for each track
            estimator: DangerEstimator for position info
            
        Returns:
            Frame with visualizations
        """
        frame_vis = frame.copy()
        
        for track, danger in zip(tracks, danger_levels):
            # Select color based on danger
            if danger == 'CRITICAL':
                color = Visualizer.COLOR_CRITICAL
            elif danger == 'WARNING':
                color = Visualizer.COLOR_WARNING
            else:
                color = Visualizer.COLOR_SAFE
            
            # Draw bounding box
            x1, y1, x2, y2 = track.bbox.astype(int)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and info
            label_parts = [
                f"ID:{track.track_id}",
                f"{track.class_name}",
                f"TTC:{track.ttc:.1f}s" if track.ttc < 999 else "TTC:--"
            ]
            
            # Position
            if estimator:
                position = estimator.get_position(track)
                label_parts.append(f"{position}")
            
            label = " | ".join(label_parts)
            
            # Background for text
            (w_text, h_text), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame_vis,
                (x1, y1 - h_text - 8),
                (x1 + w_text + 4, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                frame_vis,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
            # Draw trajectory (last 10 positions)
            if len(track.bbox_history) > 1:
                centers = []
                for bbox in list(track.bbox_history)[-10:]:
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    centers.append((cx, cy))
                
                for i in range(len(centers) - 1):
                    cv2.line(frame_vis, centers[i], centers[i+1], color, 2)
        
        return frame_vis
    
    @staticmethod
    def draw_alerts(frame: np.ndarray, alerts: List[Dict]) -> np.ndarray:
        """Draw alert banners"""
        frame_vis = frame.copy()
        
        if not alerts:
            return frame_vis
        
        h, w = frame.shape[:2]
        
        # Sort by level (CRITICAL first)
        alerts_sorted = sorted(
            alerts,
            key=lambda x: 0 if x['level'] == 'CRITICAL' else 1
        )
        
        y_offset = 30
        
        for alert in alerts_sorted[:3]:  # Show max 3 alerts
            level = alert['level']
            position = alert.get('position', 'UNKNOWN')
            ttc = alert.get('ttc', 0)
            vehicle = alert.get('class_name', 'vehicle')
            
            # Color
            if level == 'CRITICAL':
                color = Visualizer.COLOR_CRITICAL
                text = f"⚠ CRITICAL: {vehicle.upper()} approaching {position}"
            else:
                color = Visualizer.COLOR_WARNING
                text = f"⚠ WARNING: {vehicle} in {position} blind spot"
            
            if ttc < 999:
                text += f" (TTC: {ttc:.1f}s)"
            
            # Banner background
            (w_text, h_text), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(
                frame_vis,
                (10, y_offset - h_text - 10),
                (10 + w_text + 20, y_offset + 10),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                frame_vis,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                Visualizer.COLOR_TEXT,
                2
            )
            
            y_offset += h_text + 30
        
        return frame_vis
    
    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float, latency_ms: float) -> np.ndarray:
        """Draw FPS and latency info"""
        frame_vis = frame.copy()
        h, w = frame.shape[:2]
        
        text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms"
        
        # Bottom right corner
        (w_text, h_text), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        x = w - w_text - 20
        y = h - 20
        
        # Background
        cv2.rectangle(
            frame_vis,
            (x - 10, y - h_text - 10),
            (x + w_text + 10, y + 10),
            (0, 0, 0),
            -1
        )
        
        # Text
        cv2.putText(
            frame_vis,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Visualizer.COLOR_TEXT,
            2
        )
        
        return frame_vis
    
    @staticmethod
    def create_debug_view(frame: np.ndarray, tracks: List, danger_levels: List[str],
                         alerts: List[Dict], fps: float, latency_ms: float,
                         show_roi: bool = False, roi_mask: np.ndarray = None,
                         estimator=None) -> np.ndarray:
        """Create complete debug visualization"""
        frame_vis = frame.copy()
        
        # ROI
# ROI (Safe blending version)
        if show_roi and roi_mask is not None:
         h, w = frame_vis.shape[:2]

    # Ensure mask has correct size
        if roi_mask.shape[:2] != (h, w):
         roi_mask = cv2.resize(roi_mask, (w, h))

    # Ensure mask is single channel
        if len(roi_mask.shape) == 3:
           roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)

    # Convert mask to binary
        mask = (roi_mask > 0).astype(np.uint8)

    # Create colored overlay
        color_overlay = np.zeros_like(frame_vis)
        color_overlay[:, :] = (128, 0, 128)  # Purple ROI

    # Blend full frame
        blended = cv2.addWeighted(frame_vis, 0.7, color_overlay, 0.3, 0)

    # Apply only where mask == 1
        frame_vis[mask == 1] = blended[mask == 1]
        
        # Tracks
        frame_vis = Visualizer.draw_tracks(frame_vis, tracks, danger_levels, estimator)
        
        # Alerts
        frame_vis = Visualizer.draw_alerts(frame_vis, alerts)
        
        # FPS
        frame_vis = Visualizer.draw_fps(frame_vis, fps, latency_ms)
        
        return frame_vis
