"""
main.py — Person 4 (Integration)
Wires ASLClassifier → Gradio UI + macOS TTS into one app.

MediaPipe 0.10.14+ removed mp.solutions, so HandDetector (which used the
legacy API) is bypassed here. The GestureRecognizer inside ASLClassifier
already returns hand landmarks alongside each gesture result — we use those
to draw the skeleton and bounding box directly with OpenCV.

Run:
    python src/main.py
Then open http://localhost:7860 in your browser.
"""

import sys
import os

# Allow  `from models.x import ...`  and  `from ui.x import ...`
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import subprocess
import threading
from collections import deque, Counter

from models.asl_classifier import ASLClassifier
from models.asl_sign_mapper import ASLSignMapper
from ui.app import ASLGradioApp, _CSS
from utils.preprocessing import landmarks_to_bounding_box, draw_bounding_box


_say_process = None
_say_lock = threading.Lock()


def _speak_async(text: str) -> None:
    """Speak text via macOS 'say', killing any in-flight speech first."""
    global _say_process
    with _say_lock:
        if _say_process and _say_process.poll() is None:
            _say_process.terminate()
        _say_process = subprocess.Popen(
            ["say", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    _say_process.wait()


# ─────────────────────────────────────────────────────────────────────────────
# Hand skeleton drawing (replaces mp.solutions.drawing_utils)
# ─────────────────────────────────────────────────────────────────────────────

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


def draw_hand_skeleton(
    frame: np.ndarray,
    landmarks: list,
    line_color: tuple = (0, 212, 170),
    dot_color: tuple = (255, 255, 255),
    line_thickness: int = 2,
    dot_radius: int = 5,
) -> np.ndarray:
    """
    Draw the 21-point hand skeleton onto frame using OpenCV.
    landmarks: list of 21 (x, y, z) tuples with x,y normalised to [0,1].
    """
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, *_ in landmarks]

    for start, end in _HAND_CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], line_color, line_thickness, cv2.LINE_AA)

    for cx, cy in pts:
        cv2.circle(frame, (cx, cy), dot_radius, line_color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), dot_radius, dot_color, 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Gesture stabilizer — voting buffer that locks in a gesture label only when
# a majority of recent frames agree, preventing per-frame jitter.
# ─────────────────────────────────────────────────────────────────────────────

class GestureStabilizer:
    """
    Temporal voting buffer.  Only changes the displayed gesture when
    `threshold` out of the last `window` frames agree on the same label.
    """

    def __init__(self, window: int = 8, threshold: int = 5):
        self._buffer: deque = deque(maxlen=window)
        self._threshold = threshold
        self._stable_name = None
        self._stable_conf = 0.0
        self._stable_top3: list = []

    def update(self, name, confidence, top_3):
        self._buffer.append(name)
        if not self._buffer:
            return self._stable_name, self._stable_conf, self._stable_top3

        most_common, count = Counter(self._buffer).most_common(1)[0]
        if count >= self._threshold:
            self._stable_name = most_common
            self._stable_conf = confidence
            self._stable_top3 = list(top_3)

        return self._stable_name, self._stable_conf, self._stable_top3

    def reset(self):
        self._buffer.clear()
        self._stable_name = None
        self._stable_conf = 0.0
        self._stable_top3 = []


# ─────────────────────────────────────────────────────────────────────────────
# Initialise components
# ─────────────────────────────────────────────────────────────────────────────

classifier = ASLClassifier()   # loads asl_words.txt + gesture model
sign_mapper = ASLSignMapper()  # rule-based ASL sign recognition
stabilizer = GestureStabilizer(window=8, threshold=5)

# Track the last gesture spoken so we only speak once per new gesture
_last_spoken_gesture = None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — called by Gradio on every webcam frame (~10 Hz)
# ─────────────────────────────────────────────────────────────────────────────

def pipeline(frame):
    """
    frame : BGR numpy array from the webcam (via Gradio).

    Returns dict consumed by ASLGradioApp._process_live:
        {
            "frame"              : BGR ndarray  (with hand skeleton + bbox overlay),
            "prediction"         : {"sign": str, "confidence": float, "top_3": list} | None,
            "landmarks_detected" : bool,
        }
    """
    if frame is None:
        return {"frame": frame, "prediction": None, "landmarks_detected": False}

    annotated = frame.copy()

    # 1. Classify gesture — also returns 21-point landmarks from GestureRecognizer
    prediction = classifier.predict(frame=frame)
    landmarks = prediction.get("landmarks")

    # 2. Draw skeleton + bounding box when a hand is visible
    if landmarks is not None:
        draw_hand_skeleton(annotated, landmarks)

        h, w = annotated.shape[:2]
        bbox = landmarks_to_bounding_box(landmarks, w, h, padding=20)
        # Show confidence only when a sign has fired
        conf_pct = int(prediction["confidence"] * 100)
        label = f"{conf_pct}%" if conf_pct > 0 else ""
        draw_bounding_box(annotated, bbox, label=label, color=(0, 212, 170))

    # 3. Map to real ASL meaning using landmark rules + MediaPipe fallback
    raw_top3 = prediction.get("top_3", [])
    mp_gesture = raw_top3[0][0] if raw_top3 else None
    mp_conf = raw_top3[0][1] if raw_top3 else 0.0

    asl_sign = sign_mapper.classify(
        landmarks,
        mediapipe_gesture=mp_gesture,
        mediapipe_confidence=mp_conf,
    )

    if asl_sign:
        raw_name = asl_sign
        raw_conf = mp_conf if mp_conf > 0 else 0.85
        display_top3 = [(asl_sign, raw_conf)] + [
            (t[0], t[1]) for t in raw_top3[1:3]
        ] if raw_top3 else [(asl_sign, raw_conf), ("—", 0), ("—", 0)]
    elif landmarks is not None:
        raw_name = "Detecting..."
        raw_conf = 0.0
        display_top3 = []
    else:
        raw_name = None
        raw_conf = 0.0
        display_top3 = []

    # 4. Stabilize — only change when majority of recent frames agree
    stable_name, stable_conf, stable_top3 = stabilizer.update(
        raw_name, raw_conf, display_top3,
    )

    # Merge stabilized gesture info back into prediction
    stable_prediction = dict(prediction)
    stable_prediction["stable_gesture"] = stable_name
    stable_prediction["stable_confidence"] = stable_conf
    stable_prediction["stable_top3"] = stable_top3

    # 4. Auto-speak the gesture name when a NEW stable gesture is detected.
    #    Kills any in-flight speech and starts the new one immediately.
    global _last_spoken_gesture
    if (
        stable_name is not None
        and stable_name != "Detecting..."
        and stable_name != _last_spoken_gesture
    ):
        spoken = stable_name.replace("_", " ")
        _last_spoken_gesture = stable_name
        threading.Thread(
            target=_speak_async, args=(spoken,), daemon=True,
        ).start()
    elif stable_name is None:
        _last_spoken_gesture = None

    return {
        "frame":               annotated,
        "prediction":          stable_prediction,
        "landmarks_detected":  landmarks is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ASLGradioApp(use_mock=False)
    app.set_pipeline(pipeline)
    app.set_reset_fn(classifier.reset)
    app.tts_engine = None         # TTS handled via macOS 'say' in pipeline()

    demo = app.create_interface()
    demo.launch(
        css=_CSS,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False,
    )
