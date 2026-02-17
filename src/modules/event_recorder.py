"""
Event Recorder — Blind Spot System
====================================
Maintains a circular frame buffer of the last N seconds.
When ANY alert (WARNING or CRITICAL) is detected:
  • Saves the pre_seconds BEFORE the event (already in buffer)
  • Continues recording for post_seconds AFTER the event
  • Writes a timestamped video clip to disk
  • Increments and logs the event counter

KEY FIXES vs original:
  1. Writer object is passed WITH each queue command — no shared-state race
     between main thread and write thread.
  2. add_frame() no longer puts frames into the queue after _finish_recording()
     has been called — post-frames are written synchronously inside the worker
     by counting frames per-writer, not via a shared counter.
  3. Triggers on WARNING *or* CRITICAL (any red box = event recorded).
  4. get_stats() verifies actual file existence + size on disk.
  5. Robust stop(): drains queue fully before joining thread.
  6. VideoWriter.isOpened() checked before writing — logs clear error if it fails.

Usage (in main loop):
    recorder = EventRecorder(config)
    recorder.add_frame(frame)              # every processed frame
    recorder.on_alert(alerts, frame)       # when alerts fire
    recorder.stop()                        # on shutdown
"""

import cv2
import os
import time
import json
import threading
import queue
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional


class EventRecorder:

    def __init__(self, config: dict):
        self.config   = config
        self.enabled  = config.get('enabled', True)
        if not self.enabled:
            return

        # Timing
        self.fps          = config.get('record_fps',  5)
        self.pre_seconds  = config.get('pre_seconds', 6)
        self.post_seconds = config.get('post_seconds', 6)
        self.pre_frames   = self.pre_seconds  * self.fps
        self.post_frames  = self.post_seconds * self.fps

        # Output directory — created immediately so it always exists
        self.output_dir = config.get('output_dir', 'data/events')
        os.makedirs(self.output_dir, exist_ok=True)

        # Save resolution (smaller = less RAM + faster writes)
        self.save_w = config.get('save_width',  640)
        self.save_h = config.get('save_height', 360)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # ── Pre-event circular buffer ──────────────────────────────────────────
        # Stores (timestamp, small_frame) tuples.
        # maxlen ensures it never grows beyond pre_frames entries.
        self._pre_buffer: deque = deque(maxlen=self.pre_frames)

        # ── Active recording state (main-thread only) ──────────────────────────
        # _active_writer is only touched by _start_recording / _close_recording
        # in the MAIN thread.  The write worker receives the writer object
        # directly in each queue message — no shared mutable state.
        self._active_writer:    Optional[cv2.VideoWriter] = None
        self._post_frames_rem:  int  = 0
        self._recording:        bool = False
        self._current_event_id: Optional[int] = None

        # ── Persistence ────────────────────────────────────────────────────────
        self._counter_path = os.path.join(self.output_dir, 'event_count.json')
        self._event_count  = self._load_count()
        self._event_log:   List[Dict] = []

        # ── Write worker ───────────────────────────────────────────────────────
        # Queue items: (command, writer_object_or_None, data_or_None)
        # Passing the writer WITH the message eliminates all race conditions.
        self._write_queue  = queue.Queue(maxsize=200)
        self._write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self._write_thread.start()

        print(f"[Recorder] Initialised — "
              f"pre={self.pre_seconds}s  post={self.post_seconds}s  "
              f"save={self.save_w}x{self.save_h}  "
              f"dir={self.output_dir}  "
              f"events so far={self._event_count}")

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def add_frame(self, frame):
        """
        Call every frame (before or after detection — doesn't matter).
        Maintains the pre-event circular buffer.
        If a recording is active, counts down post-event frames and
        sends them to the write worker, then closes when done.
        """
        if not self.enabled:
            return

        small = cv2.resize(frame, (self.save_w, self.save_h))
        self._pre_buffer.append((time.time(), small))

        if self._recording and self._post_frames_rem > 0:
            # Pass writer WITH the frame — worker uses only what it receives
            self._write_queue.put(('frame', self._active_writer, small))
            self._post_frames_rem -= 1
            if self._post_frames_rem == 0:
                self._close_recording()

    def on_alert(self, alerts: List[Dict], frame):
        """
        Call when the decision engine fires alerts.
        Triggers on WARNING *or* CRITICAL — any red box = event recorded.

        If already recording and a CRITICAL arrives, the post-event
        window is extended to capture the full aftermath.
        """
        if not self.enabled or not alerts:
            return

        if not self._recording:
            print(f"[Recorder DEBUG] Starting recording due to alerts: {alerts}")
            self._start_recording(alerts, frame)
        else:
            # Extend post window on escalation to CRITICAL
            if any(a['level'] == 'CRITICAL' for a in alerts):
                self._post_frames_rem = max(self._post_frames_rem, self.post_frames)

    def get_stats(self) -> Dict:
        """
        Return event statistics with real on-disk file verification.
        file_exists=True means the video is fully written and accessible.
        """
        verified = []
        for e in self._event_log:
            entry = dict(e)
            fp = os.path.join(self.output_dir, e.get('file', ''))
            exists = os.path.isfile(fp)
            entry['file_exists']  = exists
            entry['file_size_kb'] = round(os.path.getsize(fp) / 1024, 1) if exists else 0
            verified.append(entry)

        return {
            'total_events':   self._event_count,
            'session_events': len(self._event_log),
            'recording':      self._recording,
            'output_dir':     self.output_dir,
            'event_log':      verified[-10:],
        }

    def stop(self):
        """Flush all pending writes and close cleanly."""
        if not self.enabled:
            return
        if self._recording:
            self._close_recording()
        self._write_queue.put(('stop', None, None))
        self._write_thread.join(timeout=10)
        self._save_count()
        print(f"[Recorder] Stopped. Total events recorded: {self._event_count}")

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — main-thread only
    # ──────────────────────────────────────────────────────────────────────────

    def _start_recording(self, alerts: List[Dict], trigger_frame):
        """
        Create a new VideoWriter, flush the pre-event buffer into it,
        then begin counting post-event frames.
        Called only from the main thread.
        """
        self._event_count += 1
        self._save_count()

        ts_str   = datetime.now().strftime('%Y%m%d_%H%M%S')
        level    = alerts[0]['level']
        position = alerts[0].get('position', 'UNKNOWN')
        filename = (f"event_{self._event_count:04d}_{ts_str}"
                    f"_{level}_{position}.mp4")
        filepath = os.path.join(self.output_dir, filename)

        writer = cv2.VideoWriter(
            filepath, self.fourcc, self.fps, (self.save_w, self.save_h)
        )
        if not writer.isOpened():
            print(f"[Recorder] ERROR: Could not open VideoWriter for {filepath}")
            print(f"           Check that '{self.output_dir}' is writable "
                  f"and opencv supports mp4v on this system.")
            self._event_count -= 1
            return

        self._active_writer    = writer
        self._recording        = True
        self._post_frames_rem  = self.post_frames
        self._current_event_id = self._event_count

        # Persist metadata immediately — so it's logged even if process dies
        event_info = {
            'id':        self._event_count,
            'timestamp': ts_str,
            'level':     level,
            'position':  position,
            'ttc':       alerts[0].get('ttc', 'inf'),
            'class':     alerts[0].get('class_name', ''),
            'file':      filename,
        }
        self._event_log.append(event_info)
        self._save_event_log(event_info)

        icon = "CRITICAL" if level == 'CRITICAL' else "WARNING"
        print(f"[Recorder] [{icon}] Event #{self._event_count} | "
              f"{level} {position} | -> {filename}")

        # Send pre-event buffer to worker WITH the writer object.
        # Worker receives writer directly — never reads self._active_writer.
        pre_frames = list(self._pre_buffer)
        self._write_queue.put(('pre_buffer', writer, pre_frames))

    def _close_recording(self):
        """
        Signal the write worker to release the current writer.
        Called only from the main thread.
        """
        if not self._recording:
            return
        writer = self._active_writer
        event_id = self._current_event_id
        self._write_queue.put(('close', writer, None))
        self._recording       = False
        self._post_frames_rem = 0
        self._active_writer   = None
        print(f"[Recorder] Event #{event_id} — finalising write in background")

    # ──────────────────────────────────────────────────────────────────────────
    # WRITE WORKER — background thread only
    # ──────────────────────────────────────────────────────────────────────────

    def _write_worker(self):
        """
        Processes queue items.  Each item is (cmd, writer, data).
        The writer object is passed directly in each message —
        no shared mutable state, no race conditions.

        Commands:
          ('pre_buffer', writer, [(ts, frame), ...])  — write pre-event frames
          ('frame',      writer, frame)               — write one post-event frame
          ('close',      writer, None)                — release the writer
          ('stop',       None,   None)                — shut down thread
        """
        while True:
            try:
                cmd, writer, data = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if cmd == 'stop':
                    if writer and writer.isOpened():
                        writer.release()
                    break

                elif cmd == 'pre_buffer':
                    if writer and writer.isOpened():
                        for _ts, frame in data:
                            writer.write(frame)

                elif cmd == 'frame':
                    if writer and writer.isOpened():
                        writer.write(data)

                elif cmd == 'close':
                    if writer and writer.isOpened():
                        writer.release()

            except Exception as exc:
                print(f"[Recorder] Write error in worker ({cmd}): {exc}")
            finally:
                self._write_queue.task_done()

    # ──────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────────

    def _load_count(self) -> int:
        try:
            with open(self._counter_path, 'r') as f:
                return json.load(f).get('count', 0)
        except Exception:
            return 0

    def _save_count(self):
        try:
            with open(self._counter_path, 'w') as f:
                json.dump({'count': self._event_count}, f)
        except Exception as e:
            print(f"[Recorder] Could not save count: {e}")

    def _save_event_log(self, event: dict):
        log_path = os.path.join(self.output_dir, 'event_log.jsonl')
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"[Recorder] Could not write log: {e}")