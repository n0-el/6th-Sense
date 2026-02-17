"""
Video Stabilization Module
Implements hybrid optical flow + homography stabilization
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple


class VideoStabilizer:
    """
    Video stabilization using optical flow and homography estimation.
    Optimized for helmet-mounted camera with high vibration.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        
        if not self.enabled:
            return
        
        # Feature detector (FAST for speed)
        self.detector = cv2.FastFeatureDetector_create()
        self.max_features = config.get('max_features', 200)
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # State
        self.prev_frame = None
        self.prev_pts = None
        
        # Temporal smoothing
        self.smoothing_window = config.get('smoothing_window', 5)
        self.transforms_buffer = deque(maxlen=self.smoothing_window)
        
        # IMU integration (if available)
        self.imu_enabled = config.get('imu', {}).get('enabled', False)
        
    def stabilize(self, frame: np.ndarray, imu_data: Optional[dict] = None) -> np.ndarray:
        """
        Stabilize a single frame.
        
        Args:
            frame: Input frame (BGR)
            imu_data: Optional IMU data (gyro, accel)
            
        Returns:
            Stabilized frame
        """
        if not self.enabled:
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        keypoints = self.detector.detect(gray, None)
        curr_pts = cv2.KeyPoint_convert(keypoints)
        
        # Limit number of points
        if len(curr_pts) > self.max_features:
            curr_pts = curr_pts[:self.max_features]
        
        curr_pts = curr_pts.astype(np.float32)
        
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            # Calculate optical flow
            curr_pts_tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, self.prev_pts, None, **self.lk_params
            )
            
            # Filter good points
            idx = np.where(status == 1)[0]
            
            if len(idx) > 10:
                prev_pts_valid = self.prev_pts[idx]
                curr_pts_tracked_valid = curr_pts_tracked[idx]
                
                # Estimate homography
                H, mask = cv2.findHomography(
                    curr_pts_tracked_valid,
                    prev_pts_valid,
                    cv2.RANSAC,
                    5.0
                )
                
                if H is not None:
                    # IMU compensation (if available)
                    if self.imu_enabled and imu_data is not None:
                        H = self._compensate_with_imu(H, imu_data)
                    
                    # Temporal smoothing
                    self.transforms_buffer.append(H)
                    H_smooth = self._smooth_transform()
                    
                    # Apply stabilization
                    h, w = frame.shape[:2]
                    frame_stabilized = cv2.warpPerspective(
                        frame, H_smooth, (w, h),
                        borderMode=cv2.BORDER_REPLICATE
                    )
                else:
                    frame_stabilized = frame
            else:
                frame_stabilized = frame
        else:
            frame_stabilized = frame
        
        # Update state
        self.prev_frame = gray.copy()
        self.prev_pts = curr_pts
        
        return frame_stabilized
    
    def _smooth_transform(self) -> np.ndarray:
        """Apply temporal smoothing to transforms"""
        if len(self.transforms_buffer) == 0:
            return np.eye(3)
        
        # Simple average (can be improved with weighted average)
        transforms = np.array(list(self.transforms_buffer))
        H_smooth = np.mean(transforms, axis=0)
        
        return H_smooth
    
    def _compensate_with_imu(self, H: np.ndarray, imu_data: dict) -> np.ndarray:
        """
        Use IMU data to compensate for known rotational motion.
        This is a simplified version - full implementation requires camera calibration.
        
        Args:
            H: Homography matrix
            imu_data: IMU readings {'gyro': [x, y, z], 'accel': [x, y, z]}
            
        Returns:
            Compensated homography
        """
        # TODO: Implement full IMU fusion with camera intrinsics
        # For now, return unchanged
        return H
    
    def reset(self):
        """Reset stabilizer state"""
        self.prev_frame = None
        self.prev_pts = None
        self.transforms_buffer.clear()
