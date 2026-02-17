"""
Decision Engine — Rear Helmet-Cam Blind Spot System
====================================================
Converts per-track danger classifications into actionable alerts.

Suppression logic tuned for rear-facing motorcycle helmet camera:
  • Traffic jam: only suppress side-zone slow vehicles, never tailgaters
  • Parked: suppress only if truly stationary for many frames AND not close
  • Lateral: do NOT suppress — lateral movement IS the overtake signal
  • Hysteresis: require 2 consistent danger frames before alerting
  • Cooldown: 1.5 s per track so alerts don't spam but do repeat if danger persists
"""

import time
import numpy as np
from collections import deque
from typing import List, Dict
from .tracking import Track


class DecisionEngine:

    def __init__(self, config: dict):
        self.config = config

        # Temporal smoothing
        self.smoothing_window      = config.get('smoothing_window', 2)  # Pi: 2 frames
        self.consistency_threshold = config.get('consistency_threshold', 1)
        self.state_history         = deque(maxlen=self.smoothing_window)

        # Alert cooldown per track
        self.cooldown_duration     = config.get('cooldown_duration', 1.5)
        self.alert_cooldown        = {}

        # Suppression flags
        self.suppress_traffic_jam  = config.get('suppress_traffic_jam', True)
        self.traffic_jam_threshold = config.get('traffic_jam_threshold', 4)
        self.suppress_parked       = config.get('suppress_parked', True)

        # NOTE: lateral suppression is DISABLED for this system.
        # Lateral movement = overtaking vehicle sweeping past — that IS the threat.

        # Hysteresis: require N consecutive danger frames before alerting
        # Set to 1 so that even a SINGLE frame with a red box fires an alert.
        self.required_frames       = config.get('required_danger_frames', 1)

        # TTC gate: don't alert if TTC > this AND proximity is also low
        self.max_ttc_for_alert     = config.get('max_ttc_for_alert', 8.0)

        # Proximity thresholds (must match danger_estimation.py)
        frame_w                    = config.get('frame_width',  1280)
        frame_h                    = config.get('frame_height',  720)
        self.frame_area            = frame_w * frame_h
        self.prox_warning          = config.get('proximity_warning',  0.06)
        self.prox_critical         = config.get('proximity_critical', 0.14)
        self.prox_slow_alert       = config.get('prox_slow_alert',    0.15)

        # Traffic jam: skip suppression for vehicles this close
        self.traffic_jam_prox_skip = config.get('traffic_jam_prox_skip', 0.06)

        # Slow traffic thresholds
        self.slow_flow_thresh      = config.get('slow_flow_thresh', 3.0)

    # ──────────────────────────────────────────────────────────────────────────

    def decide(self, tracks: List[Track], danger_levels: List[str]) -> List[Dict]:
        current_alerts = []

        for track, danger in zip(tracks, danger_levels):
            if self._is_in_cooldown(track.track_id):
                continue

            decision = self._evaluate_track(track, danger, tracks)
            if decision.get('alert', False):
                current_alerts.append(decision)
                self.alert_cooldown[track.track_id] = time.time()

        self.state_history.append(current_alerts)
        return self._find_consistent_alerts()

    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_track(self, track: Track, danger: str, all_tracks: List[Track]) -> Dict:

        bbox      = track.bbox
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        proximity = bbox_area / max(self.frame_area, 1)
        ttc       = getattr(track, 'ttc', float('inf'))
        zone      = self._get_zone(track)
        flow_mag  = getattr(track, 'flow_magnitude', 0.0)

        # ── Suppress: parked vehicle ──────────────────────────────────────────
        # Only suppress if it's been stationary for a long time AND not close.
        # A parked car at the roadside that we're about to pass is fine to ignore.
        if self.suppress_parked and self._is_parked(track) and proximity < self.prox_warning:
            print(f"[DEBUG] Suppressed PARKED: {track.track_id} (prox={proximity:.3f})")
            return {'alert': False}

        # ── Suppress: traffic jam (side zones only) ───────────────────────────
        # In slow traffic, vehicles in the SIDE zones at the same slow speed
        # are not overtaking — they're just in adjacent lanes.
        # BUT: never suppress CENTRE zone in traffic jam (tailgating is still real).
        # AND: never suppress if vehicle is already close.
        if (self.suppress_traffic_jam and
                zone in ('LEFT', 'RIGHT') and
                proximity < self.traffic_jam_prox_skip):
            slow_side = [
                t for t in all_tracks
                if t.flow_magnitude < self.slow_flow_thresh
                and self._get_zone(t) in ('LEFT', 'RIGHT')
            ]
            if len(slow_side) >= self.traffic_jam_threshold:
                print(f"[DEBUG] Suppressed JAM: {track.track_id}")
                return {'alert': False}

        # NOTE: NO lateral suppression — lateral sweep IS the overtake signal.

        # ── Hysteresis counter ────────────────────────────────────────────────
        if not hasattr(track, 'danger_counter'):
            track.danger_counter = 0

        if danger in ('CRITICAL', 'WARNING'):
            track.danger_counter += 1
        else:
            track.danger_counter = max(0, track.danger_counter - 1)

        if track.danger_counter < self.required_frames:
            # Only print if it WAS dangerous but filtered by hysteresis
            if danger in ('CRITICAL', 'WARNING'):
                print(f"[DEBUG] Suppressed HYSTERESIS: {track.track_id} ({track.danger_counter}/{self.required_frames})")
            return {'alert': False}

        # ── TTC + proximity gate ──────────────────────────────────────────────
        ttc_ok  = (ttc < float('inf') and ttc <= self.max_ttc_for_alert)
        prox_ok = (proximity >= self.prox_warning)

        if not ttc_ok and not prox_ok:
            if danger in ('CRITICAL', 'WARNING'):
               print(f"[DEBUG] Suppressed GATE: {track.track_id} (TTC={ttc:.1f}, Prox={proximity:.3f})")
            return {'alert': False}

        # ── Generate alert ────────────────────────────────────────────────────
        if danger == 'CRITICAL':
            print(f"[DEBUG] ALERT CRITICAL: {track.track_id}")
            return self._create_alert(track, 'CRITICAL', zone)

        if danger == 'WARNING':
            # Overtake (side): alert for WARNING — give the rider time to react
            if zone in ('LEFT', 'RIGHT'):
                print(f"[DEBUG] ALERT WARNING (Side): {track.track_id}")
                return self._create_alert(track, 'WARNING', zone)
            # Tailgate (centre): only fire WARNING if TTC is real or very close
            if ttc_ok or proximity >= self.prox_critical:
                print(f"[DEBUG] ALERT WARNING (Center): {track.track_id}")
                return self._create_alert(track, 'WARNING', zone)
            else:
                 print(f"[DEBUG] Suppressed CENTER WAITING: {track.track_id}")

        return {'alert': False}

    # ──────────────────────────────────────────────────────────────────────────

    def _is_in_cooldown(self, track_id: int) -> bool:
        if track_id not in self.alert_cooldown:
            return False
        return (time.time() - self.alert_cooldown[track_id]) < self.cooldown_duration

    def _is_parked(self, track: Track) -> bool:
        """True only if vehicle has been nearly stationary for many frames."""
        if track.age < 20:           # not old enough to call parked
            return False
        if len(track.flow_vectors) < 8:
            return False
        recent_flow = list(track.flow_vectors)[-8:]
        avg_flow    = np.mean([np.linalg.norm(v) for v in recent_flow])
        return avg_flow < 0.8        # basically zero movement

    def _get_zone(self, track: Track, frame_width: int = 1280) -> str:
        cx = (track.bbox[0] + track.bbox[2]) / 2.0
        if cx < frame_width * 0.33:
            return 'LEFT'
        elif cx > frame_width * 0.67:
            return 'RIGHT'
        return 'CENTER'

    def _create_alert(self, track: Track, level: str, zone: str) -> Dict:
        return {
            'alert':      True,
            'level':      level,
            'track_id':   track.track_id,
            'class_name': track.class_name,
            'ttc':        getattr(track, 'ttc', float('inf')),
            'position':   zone,
            'timestamp':  time.time(),
        }

    def _find_consistent_alerts(self) -> List[Dict]:
        """Return alerts that appeared in at least consistency_threshold recent frames."""
        if not self.state_history:
            return []

        alert_counts  = {}
        for frame_alerts in self.state_history:
            for alert in frame_alerts:
                key = (alert['track_id'], alert['level'])
                alert_counts[key] = alert_counts.get(key, 0) + 1

        consistent = []
        for alert in self.state_history[-1]:
            count = alert_counts.get((alert['track_id'], alert['level']), 0)
            if count >= self.consistency_threshold:
                consistent.append(alert)
            else:
                 print(f"[DEBUG] Dropped INCONSISTENT Alert: {alert['track_id']} ({count}/{self.consistency_threshold} frames)")
        
        return consistent

    def reset(self):
        self.state_history.clear()
        self.alert_cooldown.clear()