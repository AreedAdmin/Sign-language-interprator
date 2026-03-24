import os
import time
import numpy as np
from typing import List, Tuple, Dict, Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'models', 'gesture_recognizer.task'
)
_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'asl_words.txt'
)

# Minimum confidence for a gesture to count as "a sign was made"
CONFIDENCE_THRESHOLD = 0.5
# Seconds to wait after firing before the next word can advance
COOLDOWN_SECONDS = 1.5


class ASLClassifier:
    """
    Script-mode classifier for the demo.

    Instead of recognising which specific ASL sign was made, it advances
    sequentially through asl_words.txt each time MediaPipe detects any
    confident hand gesture. One sign = one word output.

    Usage:
        classifier = ASLClassifier()
        result = classifier.predict(frame=bgr_frame)
        # result = {'sign': 'Hi', 'confidence': 0.91, 'top_3': [...]}
        # result = {'sign': None, ...}  ← cooldown or no hand detected
    """

    def __init__(self):
        self._script = self._load_script()
        self._index = 0
        self._last_fire_time = 0.0
        self._recognizer = self._load_gesture_recognizer()
        print(f"[ASLClassifier] Script loaded: {len(self._script)} words.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Tuple[float, float, float]]] = None,
    ) -> Dict:
        """
        Call once per frame with the raw BGR frame.

        Returns the current script word if a confident gesture is detected
        and the cooldown has elapsed; otherwise returns sign=None.
        """
        no_result = {'sign': None, 'confidence': 0.0, 'top_3': [], 'index': self._index}

        if self.finished():
            return no_result

        # Enforce cooldown
        if time.time() - self._last_fire_time < COOLDOWN_SECONDS:
            return no_result

        # Detect any confident gesture in the frame
        confidence = 0.0
        top_3 = []

        if frame is not None and self._recognizer is not None:
            confidence, top_3 = self._detect_confidence(frame)
        elif landmarks is not None:
            # Fallback: any hand present counts
            confidence = 0.75
            top_3 = [('hand', 0.75)]

        if confidence < CONFIDENCE_THRESHOLD:
            return no_result

        # Fire — emit current word and advance
        word = self._script[self._index]
        self._index += 1
        self._last_fire_time = time.time()

        return {
            'sign': word,
            'confidence': confidence,
            'top_3': top_3,
            'index': self._index - 1,
        }

    def finished(self) -> bool:
        """True when all words in the script have been output."""
        return self._index >= len(self._script)

    def reset(self) -> None:
        """Restart from the beginning of the script."""
        self._index = 0
        self._last_fire_time = 0.0
        print("[ASLClassifier] Script reset.")

    def current_word(self) -> Optional[str]:
        """Peek at the next word without advancing."""
        if self.finished():
            return None
        return self._script[self._index]

    def progress(self) -> str:
        return f"{self._index}/{len(self._script)}"

    # ------------------------------------------------------------------
    # MediaPipe detection
    # ------------------------------------------------------------------

    def _detect_confidence(self, frame: np.ndarray) -> Tuple[float, list]:
        """
        Run GestureRecognizer and return (confidence, top_3).
        Any non-None gesture above threshold counts — we don't care which.
        """
        try:
            rgb = frame[:, :, ::-1].copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._recognizer.recognize(mp_image)

            if not result.gestures:
                return 0.0, []

            gestures = result.gestures[0]
            top = gestures[0]

            if top.category_name == 'None':
                return 0.0, []

            top_3 = [(g.category_name, float(g.score)) for g in gestures[:3]]
            return float(top.score), top_3

        except Exception as e:
            print(f"[ASLClassifier] Detection error: {e}")
            return 0.0, []

    def _load_gesture_recognizer(self):
        model_path = os.path.abspath(_MODEL_PATH)
        if not os.path.exists(model_path):
            print(f"[ASLClassifier] Model not found at {model_path}. "
                  "Hand-present fallback only.")
            return None
        try:
            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.GestureRecognizerOptions(
                base_options=base_opts,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            recognizer = mp_vision.GestureRecognizer.create_from_options(opts)
            print("[ASLClassifier] MediaPipe GestureRecognizer loaded.")
            return recognizer
        except Exception as e:
            print(f"[ASLClassifier] Failed to load recognizer: {e}")
            return None

    # ------------------------------------------------------------------
    # Script loading
    # ------------------------------------------------------------------

    def _load_script(self) -> List[str]:
        path = os.path.abspath(_SCRIPT_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Script not found: {path}")
        with open(path, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        return words


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------

def test_classifier():
    classifier = ASLClassifier()
    print(f"Script: {classifier._script}")
    print(f"First word: {classifier.current_word()}")

    # Simulate firing without a real frame (landmarks fallback)
    dummy_landmarks = [(0.5, 0.5, 0.0)] * 21
    result = classifier.predict(landmarks=dummy_landmarks)
    print(f"Result: {result}")

    # Second call immediately — should be blocked by cooldown
    result2 = classifier.predict(landmarks=dummy_landmarks)
    print(f"Cooldown blocked: {result2['sign'] is None}")

    print(f"Progress: {classifier.progress()}")


if __name__ == '__main__':
    test_classifier()
