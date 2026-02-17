"""
Vehicle Detection Module
YOLOv8 + Smart ROI-constrained detection for blind-spot systems

Fixes applied:
  1. ROI side zones raised (start at h*0.55 not h*0.4) to exclude pedestrian level
  2. Aspect ratio filter: rejects tall/narrow detections (people, poles)
  3. Minimum box WIDTH added (not just height) — filters pedestrian slivers
  4. Post-NMS deduplication: merge heavily overlapping boxes for same class
  5. Side-zone confidence threshold raised to reduce noisy peripheral detections
  6. Removed stray `from networkx import config` import
"""

import cv2
import numpy as np
from typing import List, Tuple
import torch
from ultralytics import YOLO


class VehicleDetector:
    """
    Real-time vehicle detection using YOLOv8.
    Optimized for blind-spot focused inference.
    """

    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    def __init__(self, config: dict):
        self.config = config
        self.model_path         = config['model_path']
        self.conf_threshold     = config['confidence_threshold']
        self.iou_threshold      = config['iou_threshold']
        self.frame_skip         = config.get('frame_skip', 3)           # working framerate
        self.use_cropped_inference = config.get('use_cropped_inference', True)
        self.crop_ratio         = config.get('crop_ratio', 0.55)        # skip top 55% (sky/trees)

        # Box size filters
        self.min_box_height     = config.get('min_box_height', 30)
        self.min_box_width      = config.get('min_box_width',  40)   # NEW: rejects thin slivers

        # Aspect ratio filter  (width / height)
        # Cars/trucks: ratio > 0.5  |  People: ratio < 0.4 (tall & narrow)
        self.min_aspect_ratio   = config.get('min_aspect_ratio', 0.40)  # NEW
        self.max_aspect_ratio   = config.get('max_aspect_ratio', 4.50)  # filter absurd shapes

        # Side-zone confidence boost
        self.side_conf_boost    = config.get('side_conf_boost', 0.10)   # extra conf needed in side zones

        # Post-NMS merge IoU
        self.merge_iou_thresh   = config.get('merge_iou_thresh', 0.45)  # NEW: merge duplicates

        self._load_model()

        self.roi_mask      = None
        self.side_mask     = None   # NEW: track which pixels are "side zones"
        self.roi_config    = config.get('roi', {})
        self.frame_count   = 0
        self.last_detections = []

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        print(f"Model loaded on {device}")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        print("Model ready!")

    # ──────────────────────────────────────────────────────────────────────────
    # ROI CREATION
    # ──────────────────────────────────────────────────────────────────────────

    def create_roi_mask(self, frame_shape: Tuple[int, int]):
        """
        Tight ROI — only the immediate danger zone around the vehicle.

        Layout (as fraction of frame):
          • Centre trapezoid : h 0.60→1.0, w 0.25→0.75  (vehicles straight ahead/behind)
          • Left  side strip : h 0.65→1.0, w 0.00→0.22  (immediate left blind spot)
          • Right side strip : h 0.65→1.0, w 0.78→1.00  (immediate right blind spot)

        Everything above h=0.60 is sky, trees, buildings, walls — excluded entirely.
        Wide side margins (>0.22 / <0.78) are footpaths/barriers — excluded.
        """
        h, w = frame_shape
        mask      = np.zeros((h, w), dtype=np.uint8)
        side_mask = np.zeros((h, w), dtype=np.uint8)

        # ── Centre danger trapezoid (tight) ──────────────────────────────────
        centre_top = int(h * 0.60)
        centre_pts = np.array([
            [int(w * 0.25), centre_top],
            [int(w * 0.75), centre_top],
            [int(w * 0.85), h         ],
            [int(w * 0.15), h         ],
        ])
        cv2.fillPoly(mask, [centre_pts], 255)

        # ── Left blind-spot strip (narrow, low) ───────────────────────────────
        left_top = int(h * 0.65)
        left_pts = np.array([
            [0,             left_top],
            [int(w * 0.20), left_top],
            [int(w * 0.22), h       ],
            [0,             h       ],
        ])
        cv2.fillPoly(mask,      [left_pts], 255)
        cv2.fillPoly(side_mask, [left_pts], 255)

        # ── Right blind-spot strip (narrow, low) ──────────────────────────────
        right_top = int(h * 0.65)
        right_pts = np.array([
            [int(w * 0.80), right_top],
            [w,             right_top],
            [w,             h        ],
            [int(w * 0.78), h        ],
        ])
        cv2.fillPoly(mask,      [right_pts], 255)
        cv2.fillPoly(side_mask, [right_pts], 255)

        self.roi_mask  = mask
        self.side_mask = side_mask

    # ──────────────────────────────────────────────────────────────────────────
    # DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[dict]:

        if self.roi_mask is None:
            self.create_roi_mask(frame.shape[:2])

        if self.frame_count % self.frame_skip != 0:
            self.frame_count += 1
            return self.last_detections

        self.frame_count += 1

        h, w = frame.shape[:2]

        # Optional cropped inference (skip sky / top half)
        if self.use_cropped_inference:
            crop_y      = int(h * self.crop_ratio)
            infer_frame = frame[crop_y:h, :]
        else:
            crop_y      = 0
            infer_frame = frame

        results = self.model(
            infer_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        raw_detections = []

        if len(results) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls  = int(boxes.cls[i].cpu().numpy())

                if cls not in self.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = xyxy
                y1 += crop_y
                y2 += crop_y

                box_h = y2 - y1
                box_w = x2 - x1

                # ── Size filters ─────────────────────────────────────────────
                if box_h < self.min_box_height:
                    continue
                if box_w < self.min_box_width:       # NEW: reject narrow slivers
                    continue

                # ── Aspect ratio filter (rejects people, poles) ───────────────
                aspect = box_w / max(box_h, 1)
                if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                    continue

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Bounds-check centre point
                cy_clamped = min(cy, h - 1)
                cx_clamped = min(cx, w - 1)

                if self.roi_mask[cy_clamped, cx_clamped] == 0:
                    continue

                # ── Side-zone confidence boost (NEW) ─────────────────────────
                # Peripheral detections must clear a higher confidence bar
                # to reduce pedestrian false positives on the edges.
                in_side = (self.side_mask[cy_clamped, cx_clamped] > 0)
                min_conf = self.conf_threshold + (self.side_conf_boost if in_side else 0.0)
                if conf < min_conf:
                    continue

                raw_detections.append({
                    'bbox':       [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class_id':   cls,
                    'class_name': self.VEHICLE_CLASSES[cls],
                    'in_side':    in_side,
                })

        # ── Post-NMS deduplication (NEW) ──────────────────────────────────────
        detections = self._merge_overlapping(raw_detections)

        self.last_detections = detections
        return detections

    # ──────────────────────────────────────────────────────────────────────────
    # POST-NMS MERGE
    # ──────────────────────────────────────────────────────────────────────────

    def _merge_overlapping(self, detections: List[dict]) -> List[dict]:
        """
        Merge heavily overlapping boxes of the same class into one.
        Keeps the highest-confidence box as representative.
        This eliminates the "5 boxes on 1 car" problem.
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        kept = []
        suppressed = [False] * len(detections)

        for i, det_i in enumerate(detections):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(detections)):
                if suppressed[j]:
                    continue
                # Only merge same class
                if detections[j]['class_id'] != det_i['class_id']:
                    continue
                if self._iou(det_i['bbox'], detections[j]['bbox']) >= self.merge_iou_thresh:
                    suppressed[j] = True

        return kept

    # ──────────────────────────────────────────────────────────────────────────
    # DRAW ROI
    # ──────────────────────────────────────────────────────────────────────────

    def draw_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.roi_mask is None:
            return frame

        overlay = frame.copy()
        h, w    = frame.shape[:2]

        if self.roi_mask.shape != (h, w):
            roi_resized = cv2.resize(self.roi_mask, (w, h))
        else:
            roi_resized = self.roi_mask

        color_overlay        = np.zeros_like(frame)
        color_overlay[:, :] = (255, 0, 255)
        blended              = cv2.addWeighted(frame, 0.8, color_overlay, 0.2, 0)

        mask = roi_resized > 0
        overlay[mask] = blended[mask]

        return overlay

    # ──────────────────────────────────────────────────────────────────────────
    # UTILS
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(bbox1: list, bbox2: list) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter == 0:
            return 0.0

        a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return inter / max(a1 + a2 - inter, 1e-6)