"""
main.py — Person 4 (Integration)
Wires ASLClassifier → Gradio UI + TTSEngine into one app.

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

from models.asl_classifier import ASLClassifier
from ui.app import ASLGradioApp, _CSS
from ui.tts_engine import TTSEngine
from utils.preprocessing import landmarks_to_bounding_box, draw_bounding_box


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
# Initialise components
# ─────────────────────────────────────────────────────────────────────────────

classifier = ASLClassifier()   # loads asl_words.txt + gesture model
tts        = TTSEngine(rate=150, volume=0.9)


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

    # 3. Speak the word the moment it fires (non-blocking)
    if prediction["sign"] is not None:
        tts.speak(prediction["sign"])

    return {
        "frame":               annotated,
        "prediction":          prediction,
        "landmarks_detected":  landmarks is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ASLGradioApp(use_mock=False)
    app.set_pipeline(pipeline)
    app.set_reset_fn(classifier.reset)
    app.tts_engine = tts          # share the same TTS instance

    demo = app.create_interface()
    demo.launch(
        css=_CSS,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False,
    )
