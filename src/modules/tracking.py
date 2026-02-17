"""
Object Tracking Module
Implements ByteTrack algorithm for multi-object tracking

Fixes applied:
  1. track_buffer: 30 → 5  — dead tracks removed after 5 missed frames (not 30)
  2. min_hits: new param (default 2) — track only reported after 2 confirmed hits,
     eliminates single-frame ghost tracks from noise detections
  3. track_thresh raised: 0.5 → 0.55 — stricter confidence to start a new track
  4. match_thresh lowered: 0.8 → 0.35 — more forgiving IoU match (was missing
     re-associations across frame_skip=3 gaps where vehicle moved further)
  5. Optical flow grid_size: 10×10=100 pts → 4×4=16 pts — 6× faster per track
  6. max_tracks cap: hard limit of 20 simultaneous tracks; oldest excess removed
  7. Active track return: only tracks with time_since_update==0 AND hits>=min_hits
"""

import time
import numpy as np
from collections import deque
from typing import List
from dataclasses import dataclass, field
import cv2


@dataclass
class Track:
    """Represents a tracked object"""
    track_id:   int
    bbox:       np.ndarray   # [x1, y1, x2, y2]
    confidence: float
    class_id:   int
    class_name: str

    # Motion history — kept short for Raspberry Pi memory + speed
    bbox_history:      deque = field(default_factory=lambda: deque(maxlen=8))
    timestamp_history: deque = field(default_factory=lambda: deque(maxlen=8))
    flow_vectors:      deque = field(default_factory=lambda: deque(maxlen=4))

    # Motion metrics
    growth_rate:    float = 0.0
    flow_magnitude: float = 0.0
    ttc:            float = float('inf')

    # Track state
    age:              int = 0
    hits:             int = 0
    time_since_update: int = 0

    # Kalman state (unused in simplified tracker)
    mean:       np.ndarray = None
    covariance: np.ndarray = None


class KalmanBoxTracker:
    """
    Simple Kalman filter for bounding box tracking.
    State: [x, y, w, h, vx, vy, vw, vh]
    """

    def __init__(self, bbox):
        x, y, w, h = self._bbox_to_xywh(bbox)
        self.kf = self._create_kalman_filter()
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).reshape((8, 1))

    def _create_kalman_filter(self):
        from filterpy.kalman import KalmanFilter
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ])
        kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ])
        kf.P *= 10.0
        kf.Q[-4:, -4:] *= 0.01
        kf.R *= 0.01
        return kf

    def predict(self):
        self.kf.predict()
        return self._xywh_to_bbox(self.kf.x[:4].flatten())

    def update(self, bbox):
        x, y, w, h = self._bbox_to_xywh(bbox)
        self.kf.update(np.array([x, y, w, h]))

    @staticmethod
    def _bbox_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _xywh_to_bbox(xywh):
        x, y, w, h = xywh
        return np.array([x, y, x + w, y + h])


class ByteTracker:
    """
    ByteTrack-style multi-object tracker, optimised for edge devices.
    """

    def __init__(self, config: dict):
        self.config = config

        # ── Tuned defaults ────────────────────────────────────────────────────
        self.track_thresh  = config.get('track_thresh',  0.55)
        self.track_buffer  = config.get('track_buffer',  2)    # 3FPS: die after 2 missed = ~0.6s
        self.match_thresh  = config.get('match_thresh',  0.25) # more forgiving at 3FPS (more movement)
        self.min_hits      = config.get('min_hits',      2)
        self.max_tracks    = config.get('max_tracks',    8)    # Pi: 8 max tracks

        self.tracks:   List[Track] = []
        self.next_id:  int         = 0
        self.prev_gray: np.ndarray = None

    # ──────────────────────────────────────────────────────────────────────────

    def update(self, detections: List[dict], frame: np.ndarray) -> List[Track]:
        """
        Update tracks with new detections.
        Returns only confirmed active tracks (hits >= min_hits, updated this frame).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Age all existing tracks
        for track in self.tracks:
            track.time_since_update += 1

        # Associate detections → tracks
        matched, unmatched_dets, _ = self._associate(detections)

        # Update matched tracks
        for det_idx, track_idx in matched:
            self._update_track(self.tracks[track_idx], detections[det_idx], gray)

        # Create new tracks for confident unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det['confidence'] >= self.track_thresh:
                self._create_track(det)

        # ── Remove dead tracks (FIX: buffer 30 → 5) ──────────────────────────
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]

        # ── Hard cap: keep only the most recently active tracks (FIX: NEW) ───
        if len(self.tracks) > self.max_tracks:
            self.tracks.sort(key=lambda t: t.time_since_update)
            self.tracks = self.tracks[:self.max_tracks]

        self.prev_gray = gray

        # Return only confirmed, currently-updated tracks (FIX: min_hits gate)
        return [
            t for t in self.tracks
            if t.time_since_update == 0 and t.hits >= self.min_hits
        ]

    # ──────────────────────────────────────────────────────────────────────────

    def _associate(self, detections: List[dict]):
        if not self.tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.tracks)))

        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            det_bbox = np.array(det['bbox'])
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = self._iou(det_bbox, track.bbox)

        matched         = []
        unmatched_dets  = list(range(len(detections)))
        unmatched_tracks= list(range(len(self.tracks)))

        while iou_matrix.size > 0 and iou_matrix.max() > self.match_thresh:
            d, t = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((unmatched_dets[d], unmatched_tracks[t]))
            unmatched_dets.pop(d)
            unmatched_tracks.pop(t)
            iou_matrix = np.delete(iou_matrix, d, axis=0)
            iou_matrix = np.delete(iou_matrix, t, axis=1)

        return matched, unmatched_dets, unmatched_tracks

    # ──────────────────────────────────────────────────────────────────────────

    def _update_track(self, track: Track, detection: dict, gray: np.ndarray):
        bbox = np.array(detection['bbox'])

        track.bbox              = bbox
        track.confidence        = detection['confidence']
        track.time_since_update = 0
        track.hits             += 1
        track.age              += 1

        track.bbox_history.append(bbox)
        track.timestamp_history.append(time.time())

        # Growth rate
        if len(track.bbox_history) >= 2:
            b_prev    = track.bbox_history[-2]
            area_prev = (b_prev[2] - b_prev[0]) * (b_prev[3] - b_prev[1])
            area_curr = (bbox[2]  - bbox[0])   * (bbox[3]  - bbox[1])
            if area_prev > 0:
                track.growth_rate = (area_curr - area_prev) / area_prev

        # Optical flow (FIX: 4×4 grid instead of 10×10)
        if self.prev_gray is not None:
            flow = self._compute_bbox_flow(gray, bbox)
            track.flow_vectors.append(flow)
            track.flow_magnitude = float(np.linalg.norm(flow))

    # ──────────────────────────────────────────────────────────────────────────

    def _create_track(self, detection: dict):
        track = Track(
            track_id   = self.next_id,
            bbox       = np.array(detection['bbox']),
            confidence = detection['confidence'],
            class_id   = detection['class_id'],
            class_name = detection['class_name'],
        )
        track.age              = 1
        track.hits             = 1
        track.time_since_update= 0
        self.tracks.append(track)
        self.next_id += 1

    # ──────────────────────────────────────────────────────────────────────────

    def _compute_bbox_flow(self, gray: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Compute average optical flow inside bbox. Uses 4×4 grid (was 10×10)."""
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.array([0.0, 0.0])

        roi_curr = gray[y1:y2, x1:x2]
        roi_prev = self.prev_gray[y1:y2, x1:x2]

        if roi_curr.size == 0 or roi_prev.size == 0:
            return np.array([0.0, 0.0])

        try:
            # FIX: 4×4 = 16 points (was 10×10 = 100 points — 6× faster)
            grid_size = 4
            y_pts = np.linspace(3, roi_curr.shape[0] - 3, grid_size)
            x_pts = np.linspace(3, roi_curr.shape[1] - 3, grid_size)
            pts   = np.array([[x, y] for y in y_pts for x in x_pts],
                             dtype=np.float32).reshape(-1, 1, 2)

            pts_new, status, _ = cv2.calcOpticalFlowPyrLK(roi_prev, roi_curr, pts, None)

            good_old = pts[status == 1]
            good_new = pts_new[status == 1]

            if len(good_old) == 0:
                return np.array([0.0, 0.0])

            return np.mean(good_new - good_old, axis=0)

        except Exception:
            return np.array([0.0, 0.0])

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter == 0:
            return 0.0

        a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = a1 + a2 - inter

        return inter / max(union, 1e-6)