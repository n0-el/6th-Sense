"""
Audio Alert System Module
Generates spatial audio alerts with escalating urgency
"""

import numpy as np
import sounddevice as sd
from scipy import signal
from typing import List, Dict, Tuple
import threading
import queue


class AudioAlertSystem:
    """
    Spatial audio alert system with directional cues and escalating urgency.
    Optimized for noisy environments (wind, engine).
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            return
        
        self.sample_rate = config.get('sample_rate', 44100)
        
        # Alert frequencies (2-4 kHz for clarity in noise)
        self.safe_freq = config.get('safe_freq', 1000)
        self.warning_freq = config.get('warning_freq', 1500)
        self.critical_freq = config.get('critical_freq', 2000)
        
        # Alert patterns
        self.safe_config = config.get('safe', {})
        self.warning_config = config.get('warning', {})
        self.critical_config = config.get('critical', {})
        
        # Spatial audio gains
        self.spatial = config.get('spatial', {})
        
        # Audio queue for non-blocking playback
        self.audio_queue = queue.Queue(maxsize=5)
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        # State
        self.last_alert_time = {}
    
    def handle_alerts(self, alerts: List[Dict]):
        """
        Process and prioritize multiple alerts.
        
        Args:
            alerts: List of alert dictionaries
        """
        if not self.enabled or not alerts:
            return
        
        # Prioritize CRITICAL over WARNING
        critical_alerts = [a for a in alerts if a['level'] == 'CRITICAL']
        warning_alerts = [a for a in alerts if a['level'] == 'WARNING']
        
        # Play at most 2 alerts to avoid cacophony
        if critical_alerts:
            self.generate_alert('CRITICAL', critical_alerts[0]['position'])
            
            if len(critical_alerts) > 1:
                # Multiple critical: Use center alert
                self.generate_alert('CRITICAL', 'CENTER')
        elif warning_alerts:
            self.generate_alert('WARNING', warning_alerts[0]['position'])
    
    def generate_alert(self, level: str, position: str):
        """
        Generate spatial audio alert.
        
        Args:
            level: Alert level ('WARNING' or 'CRITICAL')
            position: Spatial position ('LEFT', 'RIGHT', 'CENTER')
        """
        if not self.enabled:
            return
        
        # Get configuration
        if level == 'CRITICAL':
            freq = self.critical_freq
            config = self.critical_config
        elif level == 'WARNING':
            freq = self.warning_freq
            config = self.warning_config
        else:
            return  # No alert for SAFE
        
        duration = config.get('duration', 0.2)
        repetitions = config.get('repetitions', 2)
        interval = config.get('interval', 0.15)
        
        # Generate beep
        beep = self._generate_beep(freq, duration)
        
        # Spatialize
        stereo_beep = self._spatialize(beep, position)
        
        # Create alert with repetitions
        alert_signal = self._create_pattern(stereo_beep, repetitions, interval)
        
        # Queue for playback
        try:
            self.audio_queue.put_nowait(alert_signal)
        except queue.Full:
            pass  # Skip if queue full
    
    def _generate_beep(self, freq: float, duration: float) -> np.ndarray:
        """
        Generate a beep tone with envelope.
        
        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            
        Returns:
            Mono audio signal
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Generate sine wave
        beep = np.sin(2 * np.pi * freq * t)
        
        # Apply Hann window for smooth fade in/out
        envelope = signal.windows.hann(len(beep))
        beep = beep * envelope
        
        # Normalize
        beep = beep * 0.5  # Reduce volume to prevent distortion
        
        return beep
    
    def _spatialize(self, mono_signal: np.ndarray, position: str) -> np.ndarray:
        """
        Create stereo image with spatial positioning.
        
        Args:
            mono_signal: Mono audio signal
            position: 'LEFT', 'RIGHT', or 'CENTER'
            
        Returns:
            Stereo signal [N, 2]
        """
        # Get gains from config
        if position == 'LEFT':
            gains = self.spatial.get('left_gain', [1.0, 0.3])
        elif position == 'RIGHT':
            gains = self.spatial.get('right_gain', [0.3, 1.0])
        else:  # CENTER
            gains = self.spatial.get('center_gain', [0.8, 0.8])
        
        gain_left, gain_right = gains
        
        # Apply gains
        left = mono_signal * gain_left
        right = mono_signal * gain_right
        
        # Stack to stereo
        stereo = np.column_stack([left, right])
        
        return stereo
    
    def _create_pattern(self, beep: np.ndarray, repetitions: int,
                       interval: float) -> np.ndarray:
        """
        Create alert pattern with repetitions.
        
        Args:
            beep: Stereo beep signal
            repetitions: Number of repetitions
            interval: Interval between beeps in seconds
            
        Returns:
            Complete alert signal
        """
        silence = np.zeros((int(self.sample_rate * interval), 2))
        
        pattern = []
        for i in range(repetitions):
            pattern.append(beep)
            if i < repetitions - 1:
                pattern.append(silence)
        
        alert_signal = np.vstack(pattern)
        
        return alert_signal
    
    def _playback_worker(self):
        """Background thread for audio playback"""
        while True:
            try:
                # Get next alert from queue
                alert_signal = self.audio_queue.get()
                
                # Play (blocking)
                sd.play(alert_signal, self.sample_rate)
                sd.wait()
                
                self.audio_queue.task_done()
                
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def stop(self):
        """Stop audio system"""
        self.enabled = False
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass


def test_audio_system():
    """Test audio alert system"""
    config = {
        'enabled': True,
        'sample_rate': 44100,
        'critical_freq': 2000,
        'warning_freq': 1500,
        'critical': {'duration': 0.3, 'repetitions': 3, 'interval': 0.1},
        'warning': {'duration': 0.2, 'repetitions': 2, 'interval': 0.15},
        'spatial': {
            'left_gain': [1.0, 0.3],
            'right_gain': [0.3, 1.0],
            'center_gain': [0.8, 0.8]
        }
    }
    
    audio = AudioAlertSystem(config)
    
    print("Testing LEFT WARNING...")
    audio.generate_alert('WARNING', 'LEFT')
    import time
    time.sleep(1)
    
    print("Testing RIGHT CRITICAL...")
    audio.generate_alert('CRITICAL', 'RIGHT')
    time.sleep(2)
    
    print("Testing CENTER CRITICAL...")
    audio.generate_alert('CRITICAL', 'CENTER')
    time.sleep(2)
    
    audio.stop()
    print("Audio test complete!")


if __name__ == '__main__':
    test_audio_system()
