"""
Test script for individual modules
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import unittest
import numpy as np
import cv2
from modules.detection import VehicleDetector
from modules.tracking import ByteTracker, Track
from modules.danger_estimation import DangerEstimator
from modules.decision import DecisionEngine


class TestVehicleDetector(unittest.TestCase):
    """Test vehicle detection module"""
    
    def setUp(self):
        self.config = {
            'model_path': '../models/yolov8n.pt',
            'confidence_threshold': 0.4,
            'iou_threshold': 0.45,
            'frame_skip': 1,
            'use_tensorrt': False,
            'roi': {
                'rear_focus': True,
                'rear_width_ratio': 0.6
            }
        }
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = VehicleDetector(self.config)
        self.assertIsNotNone(detector.model)
    
    def test_roi_mask(self):
        """Test ROI mask creation"""
        detector = VehicleDetector(self.config)
        detector.create_roi_mask((720, 1280))
        self.assertIsNotNone(detector.roi_mask)
        self.assertEqual(detector.roi_mask.shape, (720, 1280))


class TestByteTracker(unittest.TestCase):
    """Test tracking module"""
    
    def setUp(self):
        self.config = {
            'track_thresh': 0.5,
            'track_buffer': 30,
            'match_thresh': 0.8
        }
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = ByteTracker(self.config)
        self.assertEqual(len(tracker.tracks), 0)
    
    def test_tracking(self):
        """Test basic tracking"""
        tracker = ByteTracker(self.config)
        
        # Create dummy detection
        detection = {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'class_id': 2,
            'class_name': 'car'
        }
        
        # Create dummy frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Update tracker
        tracks = tracker.update([detection], frame)
        
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 0)


class TestDangerEstimator(unittest.TestCase):
    """Test danger estimation module"""
    
    def setUp(self):
        self.config = {
            'focal_length': 800,
            'sensor_width': 6.0,
            'avg_vehicle_width': 1.8,
            'ttc_safe': 5.0,
            'ttc_warning': 3.0,
            'ttc_critical': 1.5
        }
    
    def test_initialization(self):
        """Test estimator initialization"""
        estimator = DangerEstimator(self.config)
        self.assertEqual(estimator.ttc_safe, 5.0)
    
    def test_danger_classification(self):
        """Test danger level classification"""
        estimator = DangerEstimator(self.config)
        
        # Create dummy track
        track = Track(
            track_id=0,
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name='car'
        )
        
        # Test critical
        track.ttc = 1.0
        danger = estimator.classify_danger(track)
        self.assertEqual(danger, 'CRITICAL')
        
        # Test warning
        track.ttc = 2.5
        danger = estimator.classify_danger(track)
        self.assertEqual(danger, 'WARNING')
        
        # Test safe
        track.ttc = 6.0
        danger = estimator.classify_danger(track)
        self.assertEqual(danger, 'SAFE')


class TestDecisionEngine(unittest.TestCase):
    """Test decision engine"""
    
    def setUp(self):
        self.config = {
            'smoothing_window': 3,
            'consistency_threshold': 2,
            'cooldown_duration': 2.0,
            'suppress_traffic_jam': True,
            'traffic_jam_threshold': 5
        }
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = DecisionEngine(self.config)
        self.assertEqual(engine.smoothing_window, 3)
    
    def test_alert_decision(self):
        """Test alert decision making"""
        engine = DecisionEngine(self.config)
        
        # Create dummy track
        track = Track(
            track_id=0,
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.9,
            class_id=2,
            class_name='car'
        )
        track.ttc = 1.0
        track.growth_rate = 0.15
        track.flow_magnitude = 5.0
        
        # Test decision
        alerts = engine.decide([track], ['CRITICAL'])
        
        # May not alert on first frame due to consistency requirement
        self.assertIsInstance(alerts, list)


def run_tests():
    """Run all tests"""
    print("Running module tests...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestVehicleDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestByteTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestDangerEstimator))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionEngine))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
