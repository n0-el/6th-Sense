"""
Main Blind Spot Detection System
Integrates all modules into a complete pipeline
"""

import cv2
import numpy as np
import time
import yaml
import os
from pathlib import Path
from typing import Optional

from modules.stabilization import VideoStabilizer
from modules.detection import VehicleDetector
from modules.tracking import ByteTracker
from modules.danger_estimation import DangerEstimator
from modules.decision import DecisionEngine
from modules.audio import AudioAlertSystem
from modules.event_recorder import EventRecorder
from utils.helpers import FPSTracker, PerformanceMonitor, EventLogger, Visualizer


class BlindSpotDetectionSystem:
    """
    Complete blind spot detection system for motorcycle helmets.
    """

    def __init__(self, config_path: str = '../config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("=" * 60)
        print("Blind Spot Detection System Initializing...")
        print("=" * 60)

        # Detect Docker (headless mode)
        self.headless = os.path.exists("/.dockerenv")
        if self.headless:
            print("Running in headless Docker mode (GUI disabled)")

        self._init_modules()

        self.fps_tracker = FPSTracker()

        self.perf_monitor = (
            PerformanceMonitor(self.config['logging']['metrics_file'])
            if self.config['logging']['save_metrics']
            else None
        )

        self.event_logger = (
            EventLogger(self.config['logging']['event_log'])
            if self.config['logging']['save_events']
            else None
        )

        self.frame_count = 0
        self.running = False

        print("System initialized successfully!")
        print("=" * 60)

    def _init_modules(self):
        print("Initializing video stabilization...")
        self.stabilizer = VideoStabilizer(self.config['stabilization'])

        print("Initializing vehicle detector...")
        self.detector = VehicleDetector(self.config['detection'])

        print("Initializing object tracker...")
        self.tracker = ByteTracker(self.config['tracking'])

        print("Initializing danger estimator...")
        danger_config = {
            **self.config['camera'],
            **self.config['danger_estimation']
        }
        self.estimator = DangerEstimator(danger_config)

        print("Initializing decision engine...")
        self.decision_engine = DecisionEngine(self.config['decision_engine'])

        print("Initializing audio alert system...")
        self.audio_system = AudioAlertSystem(self.config['audio_alerts'])

        print("Initializing event recorder...")
        recorder_config = {
            **self.config.get('event_recorder', {}),
            'frame_width':  self.config['camera'].get('frame_width',  1280),
            'frame_height': self.config['camera'].get('frame_height',  720),
            'record_fps':   self.config['camera'].get('fps', 10),
        }
        self.recorder = EventRecorder(recorder_config)

    def process_frame(self, frame: np.ndarray) -> dict:
        start_time = time.time()

        # Feed raw frame into recorder's pre-event circular buffer
        self.recorder.add_frame(frame)

        if self.config['stabilization']['enabled']:
            frame = self.stabilizer.stabilize(frame)

        det_start = time.time()
        detections = self.detector.detect(frame)
        det_time = (time.time() - det_start) * 1000

        track_start = time.time()
        tracks = self.tracker.update(detections, frame)
        track_time = (time.time() - track_start) * 1000

        danger_levels = []
        for track in tracks:
            ttc = self.estimator.estimate_ttc(track)
            track.ttc = ttc
            danger_levels.append(self.estimator.classify_danger(track))

        alerts = self.decision_engine.decide(tracks, danger_levels)

        # Trigger event recording and audio on any alert
        if alerts:
            self.recorder.on_alert(alerts, frame)
        self.audio_system.handle_alerts(alerts)

        total_latency = (time.time() - start_time) * 1000
        self.fps_tracker.update()
        fps = self.fps_tracker.get_fps()

        if self.perf_monitor:
            self.perf_monitor.update(det_time, track_time, total_latency, fps)

        if self.event_logger:
            self.event_logger.log_detection(len(detections), len(tracks))
            for alert in alerts:
                self.event_logger.log_alert(alert)

        self.frame_count += 1

        return {
            'frame': frame,
            'tracks': tracks,
            'danger_levels': danger_levels,
            'alerts': alerts,
            'fps': fps,
            'latency_ms': total_latency
        }

    def run(self, source: Optional[str] = None, output_path: Optional[str] = None):

        if source is None:
            source = self.config['camera']['source']

        # Handle numeric camera index properly
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        if isinstance(source, int):
            print(f"Opening camera {source}...")
            cap = cv2.VideoCapture(source)
        else:
            print(f"Opening video file: {source}")
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video resolution: {frame_width}x{frame_height}")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config['camera']['fps'],
                (frame_width, frame_height)
            )
            print(f"Saving output to: {output_path}")

        viz_config = self.config['visualization']
        self.running = True

        print("\nSystem running! Press 'q' to quit.\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or camera error")
                    break

                results = self.process_frame(frame)

                if viz_config['enabled']:
                    frame_vis = Visualizer.create_debug_view(
                        results['frame'],
                        results['tracks'],
                        results['danger_levels'],
                        results['alerts'],
                        results['fps'],
                        results['latency_ms'],
                        show_roi=viz_config['show_roi'],
                        roi_mask=self.detector.roi_mask,
                        estimator=self.estimator
                    )

                    # Save video even in headless mode
                    if writer:
                        writer.write(frame_vis)

                    # Only display if NOT Docker
                    if not self.headless:
                        cv2.imshow('Blind Spot Detection System', frame_vis)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break

                if self.frame_count % 30 == 0:
                    stats = self.recorder.get_stats()
                    rec_indicator = " 🔴REC" if stats['recording'] else ""
                    print(
                        f"Frame {self.frame_count:>6d} | "
                        f"FPS: {results['fps']:>5.1f} | "
                        f"Latency: {results['latency_ms']:>5.1f}ms | "
                        f"Tracks: {len(results['tracks']):>2d} | "
                        f"Alerts: {len(results['alerts']):>2d} | "
                        f"Events: {stats['total_events']:>3d}"
                        f"{rec_indicator}"
                    )

        finally:
            print("\nCleaning up...")
            cap.release()

            if writer:
                writer.release()

            if not self.headless:
                cv2.destroyAllWindows()

            self.audio_system.stop()
            self.recorder.stop()

            stats = self.recorder.get_stats()
            print(f"\nTotal danger events recorded: {stats['total_events']}")
            print(f"Event clips saved to: {stats.get('output_dir', 'data/events')}")
            if stats['event_log']:
                print("Last events:")
                for e in stats['event_log']:
                    file_status = (
                        f"✅ {e.get('file_size_kb', 0)} KB"
                        if e.get('file_exists') else "⏳ writing…"
                    )
                    print(f"  #{e['id']:04d} | {e['timestamp']} | "
                          f"{e['level']} {e['position']} | "
                          f"TTC={e['ttc']} | {e['class']} | "
                          f"{e['file']} [{file_status}]")

            if self.event_logger:
                self.event_logger.save()
                print(f"Events saved to {self.config['logging']['event_log']}")

            if self.perf_monitor:
                summary = self.perf_monitor.get_summary()
                print("\n" + "=" * 60)
                print("Performance Summary:")
                print("=" * 60)
                print(f"Total frames: {summary['frames']}")
                print(f"Average FPS: {summary['avg_fps']:.2f}")
                print(f"Average latency: {summary['avg_latency_ms']:.2f} ms")
                print("=" * 60)

            print("System stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Motorcycle Helmet Blind Spot Detection System'
    )
    parser.add_argument('--config', type=str, default='../config/config.yaml')
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    system = BlindSpotDetectionSystem(args.config)
    system.run(source=args.source, output_path=args.output)


if __name__ == '__main__':
    main()