"""
Audio Alert System — Raspberry Pi Optimised
============================================
No scipy dependency. Pure numpy tone generation.
Directional cues via frequency difference (not just stereo pan):
  LEFT    → lower tone in left ear
  RIGHT   → lower tone in right ear
  CENTER  → equal both ears (tailgate)

CRITICAL → fast triple beep, high frequency
WARNING  → double beep, medium frequency
"""

import numpy as np
import threading
import queue
import time
from typing import List, Dict

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("[Audio] WARNING: 'sounddevice' not found. Audio will be disabled.")


class AudioAlertSystem:

    def __init__(self, config: dict):
        self.config      = config
        self.enabled     = config.get('enabled', True)
        if not self.enabled:
            return

        self.enabled     = config.get('enabled', True)
        if not self.enabled:
            return

        self.has_hardware = (sd is not None)
        if not self.has_hardware:
            print("[Audio] ⚠️ 'sounddevice' missing. Running in SIMULATION mode (logs only).")

        self.sample_rate = config.get('sample_rate', 22050)  # lower SR = less CPU

        # Frequencies — kept in 1.5–3 kHz range (cuts through wind/engine noise)
        self.freq_warning  = config.get('warning_freq',  1800)
        self.freq_critical = config.get('critical_freq', 2600)

        # Spatial gains per position
        # Directional: one ear gets the full tone, other ear gets a quieter
        # and slightly detuned version so rider hears the direction clearly
        # Spatial gains per position
        # Directional: one ear gets the full tone, other ear gets a quieter
        # and slightly detuned version so rider hears the direction clearly
        raw_spatial = config.get('spatial', {})
        
        # Handle "flat" config style (from config.yaml) vs "nested" style (default)
        if 'center_gain' in raw_spatial:
             self.spatial = {
                'left':   {'gain': raw_spatial.get('left_gain',   [1.0, 0.2]), 'detune': -80},
                'right':  {'gain': raw_spatial.get('right_gain',  [0.2, 1.0]), 'detune':  80},
                'center': {'gain': raw_spatial.get('center_gain', [0.8, 0.8]), 'detune':   0},
            }
        else:
            # Already nested or empty -> use defaults if empty
            defaults = {
                'left':   {'gain': [1.0, 0.2], 'detune': -80},
                'right':  {'gain': [0.2, 1.0], 'detune':  80},
                'center': {'gain': [0.8, 0.8], 'detune':   0},
            }
            self.spatial = raw_spatial if raw_spatial else defaults

        # Alert cooldown — don't repeat same level within this many seconds
        self.cooldown     = config.get('alert_cooldown', 2.0)
        self._last_alert  = 0.0

        # Non-blocking playback queue
        self._queue  = queue.Queue(maxsize=3)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ──────────────────────────────────────────────────────────────────────────

    def handle_alerts(self, alerts: List[Dict]):
        """
        Called each frame with the current alert list.
        Plays highest-priority alert only.
        """
        if not self.enabled or not alerts:
            return

        # Cooldown check — avoid beep spam
        if time.time() - self._last_alert < self.cooldown:
            return

        critical = [a for a in alerts if a['level'] == 'CRITICAL']
        warning  = [a for a in alerts if a['level'] == 'WARNING']

        if critical:
            self.generate_alert('CRITICAL', critical[0]['position'])
        elif warning:
            self.generate_alert('WARNING',  warning[0]['position'])

    def generate_alert(self, level: str, position: str):
        """Generate and queue a spatial beep alert."""
        if not self.enabled:
            return

        freq  = self.freq_critical if level == 'CRITICAL' else self.freq_warning
        reps  = 3 if level == 'CRITICAL' else 2
        dur   = 0.12 if level == 'CRITICAL' else 0.18  # shorter = snappier on Pi

        signal = self._build_alert(freq, dur, reps, position)
        
        print(f"[AUDIO] 🔊 BEEP! ({level} at {position})")
        self._last_alert = time.time()

        if self.has_hardware:
            try:
                self._queue.put_nowait(signal)
            except queue.Full:
                print(f"[AUDIO] ⚠️ Audio queue full, skipping alert")
                pass   # already playing, skip

    # ──────────────────────────────────────────────────────────────────────────

    def _build_alert(self, freq: float, dur: float,
                     reps: int, position: str) -> np.ndarray:
        """Build a stereo beep pattern with directional spatial cues."""
        pos_key  = position.lower()
        if pos_key not in self.spatial:
            pos_key = 'center'
        spatial  = self.spatial[pos_key]
        gain_l, gain_r = spatial['gain']
        detune         = spatial['detune']   # Hz offset for non-dominant ear

        n_samples = int(self.sample_rate * dur)
        t         = np.linspace(0, dur, n_samples, endpoint=False)

        # Dominant ear tone
        tone_main  = np.sin(2 * np.pi * freq * t)
        # Non-dominant ear: quieter + slightly detuned (sounds clearly different)
        tone_side  = np.sin(2 * np.pi * (freq + detune) * t)

        # Simple linear fade in/out envelope (no scipy needed)
        fade       = int(n_samples * 0.1)
        envelope   = np.ones(n_samples)
        envelope[:fade]  = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)

        tone_main  = (tone_main * envelope * 0.7).astype(np.float32)
        tone_side  = (tone_side * envelope * 0.7).astype(np.float32)

        # Build stereo beep
        left  = tone_main * gain_l if gain_l >= gain_r else tone_side * gain_l
        right = tone_main * gain_r if gain_r >= gain_l else tone_side * gain_r
        beep  = np.column_stack([left, right]).astype(np.float32)

        # Silence gap between beeps
        gap_samples = int(self.sample_rate * 0.08)
        gap  = np.zeros((gap_samples, 2), dtype=np.float32)

        parts = []
        for i in range(reps):
            parts.append(beep)
            if i < reps - 1:
                parts.append(gap)

        return np.vstack(parts)

    def _worker(self):
        """Background playback thread — blocks until each sound finishes."""
        while True:
            try:
                signal = self._queue.get()
                sd.play(signal, self.sample_rate)
                sd.wait()
                self._queue.task_done()
            except Exception as e:
                print(f"[Audio] Playback error: {e}")

    def stop(self):
        self.enabled = False
        sd.stop()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                pass