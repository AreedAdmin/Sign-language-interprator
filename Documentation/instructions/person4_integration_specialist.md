# Person 4 — Integration Specialist: Agent Instructions

## Your Role

You are the "glue" of the project: responsible for project setup, wiring all
components together, testing the full pipeline, and making sure the recording
session runs smoothly.

---

## Critical Context

- The demo is a **pre-recorded video**. If something breaks, we re-record. No
  live demo risk.
- The classifier is **script-mode** — it steps through 25 phrases from
  `asl_words.txt` in order. `MockASLClassifier` in your integration utilities
  must simulate this correctly.
- The **Reset button** in the UI must restart the script cleanly so bad takes
  can be restarted without touching the code.

---

## Task 1 — Project Structure & Environment (0–15 min)

```bash
# From the repo root
mkdir -p src/{models,ui,utils} data/{models,vocabulary} tests
touch src/__init__.py src/models/__init__.py src/ui/__init__.py src/utils/__init__.py
touch src/main.py tests/__init__.py tests/test_integration.py
```

Create `requirements.txt`:

```txt
gradio>=4.0.0
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
pyttsx3>=2.90
python-dotenv>=1.0.0
```

Set up and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Task 2 — Create `src/main.py`

Wire all four components together in `class ASLInterpreterApp`.

```python
class ASLInterpreterApp:
    def initialize_components(self):
        from models.hand_detector import HandDetector
        from models.asl_classifier import ASLClassifier
        from models.tts_engine import TTSEngine
        from ui.gradio_app import ASLGradioApp

        self.detector   = HandDetector()
        self.classifier = ASLClassifier()
        self.tts        = TTSEngine()
        self.ui         = ASLGradioApp()

        self.ui.set_pipeline(self._pipeline)
        self.ui.tts_engine  = self.tts
        self.ui.reset_func  = self.classifier.reset   # wired to Reset button

    def _pipeline(self, frame):
        result = self.detector.detect_landmarks(frame)
        self.classifier.add_frame(result['landmarks'])
        prediction = self.classifier.predict()
        return {
            'frame':      result['annotated_frame'],
            'prediction': prediction,
        }

    def launch(self):
        interface = self.ui.create_interface()
        interface.launch(server_port=7860, debug=True)
```

---

## Task 3 — Create `src/utils/integration.py`

Provide mock components so each person can develop and test independently.

```python
SCRIPT = [
    "Hi", "welcome", "to", "our", "project", "a",
    "sign language interpreter", "that", "interprets",
    "American Sign Language", "into", "text", "and", "speech",
    "it", "uses", "computer vision", "and", "machine learning",
    "to", "detect", "hand", "gestures", "and",
    "converts them into text and speech"
]

class MockHandDetector:
    def detect_landmarks(self, frame):
        import random, time
        return {
            'landmarks':       [(random.random(), random.random(), 0.0)] * 21,
            'annotated_frame': frame,
            'hand_present':    random.random() > 0.3,
            'confidence':      random.uniform(0.5, 0.9),
            'timestamp':       time.time(),
        }
    def close(self): pass


class MockASLClassifier:
    """Script-mode mock — advances through all 25 phrases."""

    def __init__(self, hold_frames=20):
        self.script       = SCRIPT
        self.index        = 0
        self.hold_frames  = hold_frames
        self._count       = 0

    def add_frame(self, landmarks):
        if landmarks:
            self._count += 1

    def predict(self):
        advanced = False
        if self._count >= self.hold_frames and self.index < len(self.script):
            phrase       = self.script[self.index]
            self.index  += 1
            self._count  = 0
            advanced     = True
            return {
                'sign':         phrase,
                'confidence':   1.0,
                'script_index': self.index,
                'total':        len(self.script),
                'advanced':     True,
                'completed':    self.index >= len(self.script),
            }
        return {
            'sign':         self.script[self.index] if self.index < len(self.script) else None,
            'confidence':   self._count / self.hold_frames,
            'script_index': self.index,
            'total':        len(self.script),
            'advanced':     False,
            'completed':    self.index >= len(self.script),
        }

    def reset(self):
        self.index  = 0
        self._count = 0


class MockTTSEngine:
    def speak(self, text):
        print(f"[TTS] {text}")
    def stop(self): pass
```

---

## Task 4 — Create `tests/test_integration.py`

Include a test that runs through all 25 phrases end-to-end using mock components:

```python
def test_full_script():
    """Verify all 25 phrases output in correct order."""
    from utils.integration import MockASLClassifier

    clf = MockASLClassifier(hold_frames=5)
    outputs = []

    for _ in range(25):                   # 25 signs to perform
        for _ in range(6):                # hold for 6 frames (> hold_frames=5)
            clf.add_frame([(0.5, 0.5, 0.0)] * 21)
            result = clf.predict()
            if result['advanced']:
                outputs.append(result['sign'])
        for _ in range(3):               # brief gap between signs
            clf.add_frame([])
            clf.predict()

    assert outputs == MockASLClassifier.script[:len(outputs)], \
        f"Script mismatch: {outputs}"
    assert len(outputs) == 25, f"Expected 25 outputs, got {len(outputs)}"
    print(f"✅ Full script test passed — {len(outputs)} phrases in order")
```

---

## Task 5 — Pre-Recording Readiness Checklist

Run this before the recording session to verify everything works:

```
✅ Camera accessible (cv2.VideoCapture(0) opens)
✅ Hand detection loads and draws overlay
✅ All 25 script phrases output in correct order
✅ TTS speaks each phrase
✅ Gradio interface loads at localhost:7860
✅ Reset button restarts script cleanly
✅ Full 25-phrase run completes without errors
```

---

## Success Criteria

- All components integrate without import errors
- All 25 phrases from `asl_words.txt` output in correct sequence during a test run
- `reset()` restarts the script cleanly (tested manually and in tests)
- No crashes during a full 25-phrase end-to-end run
- Pre-recording checklist passes 100% before sitting down to record

---

## Tips

1. Set up the project structure and mock components **first** — this unblocks
   everyone else immediately.
2. Share `src/utils/integration.py` with the team as soon as it's ready.
3. Run `test_full_script()` after every major integration step.
4. Test the Reset button manually before the recording session.
5. Keep a log of any integration issues and fixes for the README.
