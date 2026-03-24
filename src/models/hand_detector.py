"""
hand_detector.py — Person 2 (Fares) — Computer Vision
MediaPipe-based hand landmark detection with temporal buffering,
smoothing, and frame-skip optimisation for real-time performance.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────────────────────
# Landmark index constants (MediaPipe 21-point hand model)
# ─────────────────────────────────────────────────────────────
WRIST         = 0
THUMB_CMC     = 1;  THUMB_MCP     = 2;  THUMB_IP  = 3;  THUMB_TIP  = 4
INDEX_MCP     = 5;  INDEX_PIP     = 6;  INDEX_DIP = 7;  INDEX_TIP  = 8
MIDDLE_MCP    = 9;  MIDDLE_PIP    = 10; MIDDLE_DIP= 11; MIDDLE_TIP = 12
RING_MCP      = 13; RING_PIP      = 14; RING_DIP  = 15; RING_TIP   = 16
PINKY_MCP     = 17; PINKY_PIP     = 18; PINKY_DIP = 19; PINKY_TIP  = 20


class HandDetector:
    """
    Real-time hand landmark detector wrapping MediaPipe Hands.

    Responsibilities (Person 2):
      • Capture / receive BGR frames
      • Detect up to 1 hand and extract 21 normalised landmarks
      • Annotate frames for live display
      • Buffer recent landmarks and expose smoothed / temporal views
      • Frame-skip optimisation for ≥10 FPS on CPU
      • Expose clean output dict consumed by Person 1's ASLClassifier
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        buffer_size: int = 5,
        skip_frames: int = 0,
    ):
        """
        Args:
            max_num_hands:              Max simultaneous hands (keep 1 for speed).
            min_detection_confidence:   Threshold for initial hand detection.
            min_tracking_confidence:    Threshold for subsequent tracking frames.
            buffer_size:                How many recent frames to keep for smoothing.
            skip_frames:                Number of frames to skip between full detections
                                        (0 = process every frame; 2 = process every 3rd).
        """
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # ── Temporal buffers ─────────────────────────────────
        self.buffer_size     = buffer_size
        self.landmark_buffer = deque(maxlen=buffer_size)   # None when no hand
        self.confidence_buffer = deque(maxlen=buffer_size)

        # ── Performance / statistics ──────────────────────────
        self.skip_frames          = skip_frames
        self._skip_counter        = 0
        self._last_result: Optional[Dict] = None

        self.frame_count     = 0
        self.detection_count = 0
        self.total_latency   = 0.0
        self._start_time     = time.time()

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Dict:
        """
        Main entry point.  Call once per webcam frame.

        Args:
            frame:  BGR uint8 array from cv2.VideoCapture.

        Returns:
            {
              'hand_present':        bool,
              'landmarks':           List[(x,y,z)] × 21  or  None,
              'smoothed_landmarks':  List[(x,y,z)] × 21  or  None,
              'temporal_landmarks':  List[List[(x,y,z)]] (buffer) or None,
              'annotated_frame':     BGR ndarray with skeleton drawn,
              'confidence':          float 0–1,
              'timestamp':           float (epoch),
              'frame_id':            int,
            }
        """
        t0 = time.perf_counter()
        self.frame_count += 1

        # ── Frame-skip optimisation ───────────────────────────
        if self.skip_frames > 0 and self._skip_counter < self.skip_frames:
            self._skip_counter += 1
            if self._last_result is not None:
                recycled = dict(self._last_result)
                recycled["annotated_frame"] = frame.copy()
                recycled["frame_id"] = self.frame_count
                recycled["timestamp"] = time.time()
                return recycled

        self._skip_counter = 0

        # ── MediaPipe inference ───────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            self.detection_count += 1
            hand_lm = results.multi_hand_landmarks[0]

            landmarks  = self._extract_coordinates(hand_lm)
            confidence = self._estimate_confidence(hand_lm)

            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                annotated,
                hand_lm,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Update buffers
            self.landmark_buffer.append(landmarks)
            self.confidence_buffer.append(confidence)

            result = {
                "hand_present":       True,
                "landmarks":          landmarks,
                "smoothed_landmarks": self._smoothed_landmarks(),
                "temporal_landmarks": self._temporal_landmarks(),
                "annotated_frame":    annotated,
                "confidence":         confidence,
                "timestamp":          time.time(),
                "frame_id":           self.frame_count,
            }
        else:
            self.landmark_buffer.append(None)
            self.confidence_buffer.append(0.0)

            result = {
                "hand_present":       False,
                "landmarks":          None,
                "smoothed_landmarks": None,
                "temporal_landmarks": None,
                "annotated_frame":    annotated,
                "confidence":         0.0,
                "timestamp":          time.time(),
                "frame_id":           self.frame_count,
            }

        latency = time.perf_counter() - t0
        self.total_latency += latency
        self._last_result = result
        return result

    def get_statistics(self) -> Dict:
        """Return running performance statistics."""
        elapsed   = time.time() - self._start_time
        avg_lat   = (self.total_latency / self.frame_count * 1000) if self.frame_count else 0
        det_rate  = (self.detection_count / self.frame_count) if self.frame_count else 0
        fps       = self.frame_count / elapsed if elapsed > 0 else 0

        return {
            "total_frames":     self.frame_count,
            "total_detections": self.detection_count,
            "detection_rate":   round(det_rate, 3),
            "avg_latency_ms":   round(avg_lat, 2),
            "approx_fps":       round(fps, 1),
            "elapsed_seconds":  round(elapsed, 1),
        }

    def reset_statistics(self):
        """Reset all counters and timers."""
        self.frame_count     = 0
        self.detection_count = 0
        self.total_latency   = 0.0
        self._start_time     = time.time()

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    def _extract_coordinates(
        self, hand_landmarks
    ) -> List[Tuple[float, float, float]]:
        """
        Return 21 (x, y, z) tuples.
        x, y are in [0, 1] (normalised to frame size).
        z is depth relative to wrist (negative = closer to camera).
        """
        return [
            (lm.x, lm.y, lm.z)
            for lm in hand_landmarks.landmark
        ]

    def _estimate_confidence(self, hand_landmarks) -> float:
        """
        Proxy confidence: fraction of landmarks whose visibility > 0.
        MediaPipe always sets visibility to 0 for hand landmarks,
        so we fall back to checking that coordinates are inside [0,1].
        """
        inside = sum(
            1 for lm in hand_landmarks.landmark
            if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0
        )
        return inside / len(hand_landmarks.landmark)

    def _smoothed_landmarks(self) -> Optional[List[Tuple[float, float, float]]]:
        """Average the last N valid (non-None) frames in the buffer."""
        valid = [lm for lm in self.landmark_buffer if lm is not None]
        if not valid:
            return None

        smoothed = []
        for i in range(21):
            xs = [f[i][0] for f in valid]
            ys = [f[i][1] for f in valid]
            zs = [f[i][2] for f in valid]
            n  = len(valid)
            smoothed.append((sum(xs)/n, sum(ys)/n, sum(zs)/n))
        return smoothed

    def _temporal_landmarks(
        self,
    ) -> Optional[List[List[Tuple[float, float, float]]]]:
        """Return the full landmark buffer only when it is completely filled."""
        if len(self.landmark_buffer) < self.buffer_size:
            return None
        if any(lm is None for lm in self.landmark_buffer):
            return None
        return list(self.landmark_buffer)


# ─────────────────────────────────────────────────────────────
# Standalone webcam test  (python -m src.models.hand_detector)
# ─────────────────────────────────────────────────────────────
def run_webcam_test():
    """
    Quick smoke-test.  Press:
      q  – quit
      s  – print statistics
      r  – reset statistics
    """
    import os
    cam_index = int(os.getenv("CAMERA_INDEX", "0"))
    detector  = HandDetector(skip_frames=0)
    cap       = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cam_index}. "
              "Set CAMERA_INDEX in .env to the correct device index.")
        return

    print("HandDetector test — press  q=quit  s=stats  r=reset")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        result = detector.detect(frame)
        display = result["annotated_frame"].copy()

        # ── HUD overlay ───────────────────────────────────────
        status = "Hand detected" if result["hand_present"] else "No hand"
        color  = (0, 220, 60) if result["hand_present"] else (0, 80, 220)
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        conf_text = f"Conf: {result['confidence']:.2f}"
        cv2.putText(display, conf_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        stats = detector.get_statistics()
        fps_text = f"FPS: {stats['approx_fps']}"
        cv2.putText(display, fps_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("HandDetector — Person 2 (Fares)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print("Statistics:", detector.get_statistics())
        elif key == ord("r"):
            detector.reset_statistics()
            print("Statistics reset.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Test complete.")
    print("Final statistics:", detector.get_statistics())


if __name__ == "__main__":
    run_webcam_test()
