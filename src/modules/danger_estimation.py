"""
Danger Estimation Module — Rear Helmet-Cam Blind Spot System
=============================================================
Camera: Rear-facing, mounted on motorcycle helmet.

Threat scenarios:
  1. OVERTAKE  — vehicle entering LEFT or RIGHT zone and growing (approaching
                 from behind to overtake through the blind spot)
  2. TAILGATE  — vehicle in CENTRE zone and growing (closing from directly behind)

What must NOT alert:
  • Vehicles we are pulling away from      → bbox shrinking
  • Vehicles at the same speed as us       → bbox stable, no growth
  • Slow / stopped traffic                 → only alert if proximity > 15%
  • Pillion rider at bottom-centre         → always present, never a threat
  • Head-tilt / road-bump vibration        → ego-motion compensation
  • Parallel vehicles moving at same speed → lateral flow only, no growth
"""

import numpy as np
from .tracking import Track


# ── Proximity thresholds (bbox_area / frame_area) ─────────────────────────────
# Rear helmet-cam: a car 5 m behind fills ~6–8% of frame
_PROX_MONITOR   = 0.01   # ANY visible box → start tracking (was 0.03)
_PROX_WARNING   = 0.04   # getting close (was 0.06) — earlier warning
_PROX_CRITICAL  = 0.12   # very close — alert regardless of speed (was 0.14)

# In slow/stopped traffic, only alert at this proximity (basically touching)
_PROX_SLOW_ALERT = 0.12  # was 0.15 — slightly more sensitive in slow traffic

# ── Growth thresholds ─────────────────────────────────────────────────────────
_GROWTH_THRESH       = 0.002   # was 0.004 — half the old threshold, catches slower approaches
_SHRINK_THRESH       = -0.002  # negative growth = pulling away → SAFE
_SMOOTHING_N         = 3       # was 6 — 3 frames is enough at Pi's low FPS
_ALPHA               = 0.45    # EMA weight for TTC smoothing

# ── Overtake lateral entry signal ─────────────────────────────────────────────
# Overtaking vehicle sweeps in laterally AND grows simultaneously
_LATERAL_ENTRY_FLOW  = 3.0     # px/frame lateral flow component

# ── Pillion / self-exclusion zone ─────────────────────────────────────────────
# Pillion's back is always visible at bottom-centre, very large, never moves
_PILLION_Y_FRAC      = 0.82    # below 82% of frame height
_PILLION_X_LO        = 0.28    # horizontally centred between 28–72%
_PILLION_X_HI        = 0.72
_PILLION_PROX        = 0.18    # AND occupies >18% of frame → it's us

# ── Ego-motion (head tilt / road vibration) ───────────────────────────────────
_EGO_SHIFT_THRESH    = 10.0    # px/frame — if above, subtract from growth


class DangerEstimator:
    """
    Blind-spot overtake + tailgate danger estimator for rear helmet-cam.
    """

    def __init__(self, config: dict):
        self.config = config

        # TTC thresholds (seconds) — tighter than generic system for bike safety
        self.ttc_critical        = config.get('ttc_critical', 2.5)
        self.ttc_warning         = config.get('ttc_warning',  5.0)

        # Motion
        self.min_growth_rate     = config.get('min_growth_rate',    _GROWTH_THRESH)
        self.min_flow_magnitude  = config.get('min_flow_magnitude',  1.5)
        self.fps                 = config.get('fps', 30.0)

        # Smoothing
        self.alpha               = config.get('ttc_smoothing_alpha', _ALPHA)
        self.smoothing_n         = config.get('ttc_smoothing_n',     _SMOOTHING_N)

        # Frame dimensions
        self.frame_w             = config.get('frame_width',  1280)
        self.frame_h             = config.get('frame_height',  720)

        # Motorcycle class gets tighter TTC (faster + narrower = less warning time)
        self.motorcycle_ttc_mult = config.get('motorcycle_ttc_multiplier', 0.75)

        # Ego-motion — updated each frame via update_ego_flow()
        self.ego_flow            = np.array([0.0, 0.0])
        self.ego_shift_thresh    = config.get('ego_shift_thresh', _EGO_SHIFT_THRESH)

        # Slow traffic thresholds
        self.slow_ego_thresh     = config.get('slow_ego_thresh',  4.0)  # px/frame
        self.slow_rel_thresh     = config.get('slow_rel_thresh',  3.0)  # px/frame

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def update_ego_flow(self, ego_flow: np.ndarray):
        """
        Called each frame with the estimated global camera motion vector
        (dx, dy) in pixels/frame. Accounts for head tilts and road vibration.
        Comes from the VideoStabilizer or a background-flow estimate.
        """
        self.ego_flow = np.asarray(ego_flow, dtype=float)

    def estimate_ttc(self, track: Track) -> float:
        """
        Estimate Time-to-Collision in seconds. Returns inf if not approaching.
        Updates track.ttc in-place.
        """
        ttc = self._growth_based_ttc(track)

        # Fallback: optical flow expansion if growth is below threshold
        if ttc == float('inf') and len(track.flow_vectors) >= 2:
            ttc = self._flow_based_ttc(track)

        # Motorcycles have less margin — tighten effective TTC
        if ttc < float('inf') and track.class_name == 'motorcycle':
            ttc *= self.motorcycle_ttc_mult

        # EMA smoothing — hold last valid estimate across a missed frame
        if ttc < float('inf'):
            prev = getattr(track, 'smoothed_ttc', float('inf'))
            track.smoothed_ttc = (
                self.alpha * prev + (1.0 - self.alpha) * ttc
                if prev < float('inf') else ttc
            )
            track.ttc = track.smoothed_ttc
        else:
            if not hasattr(track, 'smoothed_ttc'):
                track.smoothed_ttc = float('inf')
            track.ttc = track.smoothed_ttc

        print(f"[TTC] ID:{track.track_id:3d} {track.class_name:12s} | "
              f"TTC={track.ttc:6.2f}s | ego={np.linalg.norm(self.ego_flow):.1f}px")
        return track.ttc

    def classify_danger(self, track: Track) -> str:
        """
        Classify danger for a rear-facing blind spot camera.

        Two real threats:
          OVERTAKE — side zone vehicle, same direction, growing steadily
          TAILGATE — centre zone vehicle, same direction, closing fast

        Two false alarm sources fixed here:
          OPPOSITE DIRECTION — grows fast then shrinks, high lateral speed,
                               passes through frame quickly → suppress
          PROXIMITY-ONLY CRITICAL — large bbox alone is NOT enough for CRITICAL,
                               must also be growing (actually approaching)
        """
        ttc          = getattr(track, 'ttc', float('inf'))
        bbox         = track.bbox
        zone         = self._get_zone(track)
        flow_mag     = getattr(track, 'flow_magnitude', 0.0)

        bbox_area    = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        proximity    = bbox_area / max(self.frame_w * self.frame_h, 1)

        cx_frac      = ((bbox[0] + bbox[2]) / 2.0) / self.frame_w
        cy_frac      = ((bbox[1] + bbox[3]) / 2.0) / self.frame_h

        recent_growth = self._recent_avg_growth(track)
        lateral_flow  = (abs(track.flow_vectors[-1][0])
                         if track.flow_vectors else 0.0)
        forward_flow  = (track.flow_vectors[-1][1]
                         if track.flow_vectors else 0.0)

        # ── Guard 1: Pillion / own body ───────────────────────────────────────
        if (cy_frac       >= _PILLION_Y_FRAC and
                _PILLION_X_LO <= cx_frac <= _PILLION_X_HI and
                proximity     >= _PILLION_PROX):
            print(f"[CLS] ID:{track.track_id:3d} | PILLION/SELF excluded")
            return 'SAFE'

        # ── Guard 2: Opposite direction vehicle ───────────────────────────────
        # Opposite-direction vehicles pass through the frame very quickly.
        # Signature: HIGH lateral flow, bbox peaks then immediately shrinks,
        # track age is short (they're gone in a few frames).
        # We detect this by checking if the vehicle is MOVING AWAY now
        # (shrinking) after a brief appearance, OR has very high lateral
        # flow with short track history.
        is_opposite = self._is_opposite_direction(track, lateral_flow,
                                                   recent_growth, flow_mag)
        if is_opposite:
            print(f"[CLS] ID:{track.track_id:3d} | OPPOSITE DIRECTION, suppressed")
            return 'SAFE'

        # ── Guard 3: Vehicle shrinking → we're pulling away ───────────────────
        if recent_growth < _SHRINK_THRESH:
            print(f"[CLS] ID:{track.track_id:3d} | SHRINKING ({recent_growth:.4f}), safe")
            return 'SAFE'

        # ── Guard 4: Slow / stopped traffic ───────────────────────────────────
        ego_speed       = float(np.linalg.norm(self.ego_flow))
        in_slow_traffic = (ego_speed < self.slow_ego_thresh and
                           flow_mag  < self.slow_rel_thresh)
        if in_slow_traffic:
            if proximity < _PROX_SLOW_ALERT:
                print(f"[CLS] ID:{track.track_id:3d} | SLOW TRAFFIC suppressed")
                return 'SAFE'
            return 'WARNING'   # extremely close even in slow traffic

        # ── Determine if vehicle is actually approaching ───────────────────────
        # CRITICAL requires BOTH proximity AND confirmed approach (growth or TTC).
        # Proximity alone (large bbox) is only WARNING — it might be a vehicle
        # passing at speed in the adjacent lane, not closing on us.
        is_approaching = (recent_growth > self.min_growth_rate or
                          ttc < float('inf'))

        # ── Zone: LEFT or RIGHT — Overtake detection ──────────────────────────
        if zone in ('LEFT', 'RIGHT'):
            # Same-direction overtake: grows from behind into side zone
            # A vehicle simply being in the side zone at monitor proximity
            # is worth a WARNING — the rider should be aware of it.
            overtake_signal = (recent_growth > self.min_growth_rate or
                               lateral_flow  > _LATERAL_ENTRY_FLOW or
                               proximity     >= _PROX_MONITOR)  # ANY visible box
            if not overtake_signal:
                return 'SAFE'

            # CRITICAL: close AND confirmed approach
            if proximity >= _PROX_CRITICAL and is_approaching:
                if ttc <= self.ttc_critical:
                    return 'CRITICAL'
                # Close but TTC not that tight yet → WARNING
                return 'WARNING'

            # WARNING: moderate proximity with approach, OR tight TTC
            if ttc <= self.ttc_critical:
                return 'CRITICAL'
            if ttc <= self.ttc_warning or proximity >= _PROX_WARNING:
                return 'WARNING'
            # Even at monitor proximity, flag as WARNING so it's recorded
            if proximity >= _PROX_MONITOR:
                return 'WARNING'
            return 'SAFE'

        # ── Zone: CENTRE — Tailgate detection ────────────────────────────────
        # CRITICAL requires confirmed approach, not just proximity
        if ttc <= self.ttc_critical and is_approaching:
            return 'CRITICAL'
        if proximity >= _PROX_CRITICAL and is_approaching:
            return 'CRITICAL'
        if ttc <= self.ttc_warning or proximity >= _PROX_WARNING:
            return 'WARNING'
        # Any detected vehicle in the centre zone (even far) = WARNING
        # so it gets recorded and the rider is aware
        if proximity >= _PROX_MONITOR:
            return 'WARNING'
        return 'SAFE'

    def _is_opposite_direction(self, track: Track, lateral_flow: float,
                                recent_growth: float, flow_mag: float) -> bool:
        """
        Detect vehicles coming in the opposite direction (towards camera).

        Opposite-direction signature:
          1. Very high lateral flow (they sweep across frame quickly)
          2. Short track age — they appear and disappear fast
          3. Growth pattern: grows then immediately starts shrinking
             (we only see them for 5–10 frames total)
          4. Forward optical flow is NEGATIVE (moving toward camera = away in
             rear-facing setup means they're going in the opposite direction)

        Same-direction overtaker signature:
          1. Moderate lateral flow (sweeping to pass)
          2. Sustained growth over many frames (they were behind for a while)
          3. Longer track age
        """
        # Short-lived track with very high lateral sweep = opposite direction
        if track.age < 6 and lateral_flow > 8.0:
            return True

        # High flow magnitude but no consistent growth = passing at speed
        # (opposite direction cars appear briefly, high speed, no sustained growth)
        if flow_mag > 12.0 and recent_growth < self.min_growth_rate:
            return True

        # Check historical growth pattern: grew fast then immediately shrinking
        # This is the classic "approaching then passing" signature
        bboxes = list(track.bbox_history)
        if len(bboxes) >= 4:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
            mid   = len(areas) // 2
            first_half_growth = (areas[mid] - areas[0]) / max(areas[0], 1)
            second_half_growth= (areas[-1] - areas[mid]) / max(areas[mid], 1)
            # Grew in first half, now shrinking in second half = passed us
            if first_half_growth > 0.05 and second_half_growth < -0.05:
                return True

        return False

    def get_position(self, track: Track, frame_width: int = None) -> str:
        return self._get_zone(track, frame_width)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _get_zone(self, track: Track, frame_width: int = None) -> str:
        fw = frame_width or self.frame_w
        cx = (track.bbox[0] + track.bbox[2]) / 2.0
        if cx < fw * 0.33:
            return 'LEFT'
        elif cx > fw * 0.67:
            return 'RIGHT'
        return 'CENTER'

    def _growth_based_ttc(self, track: Track) -> float:
        """
        Rolling N-frame average bbox growth → TTC.
        Subtracts ego-motion-induced apparent growth from head tilts / bumps.
        """
        bboxes = list(track.bbox_history)
        if len(bboxes) < 3:
            return float('inf')

        areas = np.array(
            [(b[2]-b[0]) * (b[3]-b[1]) for b in bboxes], dtype=float
        )
        n      = min(self.smoothing_n, len(areas) - 1)
        recent = areas[-(n+1):]
        prev_a, curr_a = recent[:-1], recent[1:]

        valid = prev_a > 0
        if not np.any(valid):
            return float('inf')

        rates      = (curr_a[valid] - prev_a[valid]) / prev_a[valid]
        avg_growth = float(np.mean(rates))

        # Head-tilt / vibration compensation:
        # A camera shift of S pixels makes a bbox appear to grow by ~S/frame_dim.
        # Subtract this estimate from the observed growth.
        ego_mag = float(np.linalg.norm(self.ego_flow))
        if ego_mag > self.ego_shift_thresh:
            ego_apparent = ego_mag / max(self.frame_w, self.frame_h)
            avg_growth   = max(0.0, avg_growth - ego_apparent)

        if avg_growth < self.min_growth_rate:
            return float('inf')

        return max(0.0, 1.0 / (avg_growth * self.fps))

    def _flow_based_ttc(self, track: Track) -> float:
        """Optical flow expansion rate as fallback TTC estimate."""
        flows = list(track.flow_vectors)
        if len(flows) < 2:
            return float('inf')

        avg_mag = float(np.mean([np.linalg.norm(v) for v in flows[-2:]]))
        if avg_mag < self.min_flow_magnitude:
            return float('inf')

        bbox_w = track.bbox[2] - track.bbox[0]
        if bbox_w <= 0:
            return float('inf')

        expansion = (avg_mag / bbox_w) * self.fps
        return max(0.0, 1.0 / expansion) if expansion > 0 else float('inf')

    def _recent_avg_growth(self, track: Track) -> float:
        """
        Average growth rate over last N frames.
        Negative = vehicle shrinking = we're pulling away.
        """
        bboxes = list(track.bbox_history)
        if len(bboxes) < 3:
            return 0.0

        areas  = [(b[2]-b[0]) * (b[3]-b[1]) for b in bboxes]
        n      = min(self.smoothing_n, len(areas) - 1)
        recent = areas[-(n+1):]
        rates  = [
            (recent[i] - recent[i-1]) / recent[i-1]
            for i in range(1, len(recent))
            if recent[i-1] > 0
        ]
        return float(np.mean(rates)) if rates else 0.0