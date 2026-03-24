# Person 1: ML Expert (Team Lead) - Tasks & Guidelines

## Role Overview
You are responsible for the core intelligence of the ASL interpreter - the machine learning model that converts hand landmarks into sign predictions, plus the text-to-speech functionality.

## Timeline & Tasks

### Hour 1: ASL Model Setup (60 minutes)

#### Task 1: Model Selection & Setup (0-15 min)
**Options (choose one):**

**Option A: Pre-trained Model (Recommended)**
```python
# Use a simple gesture recognition model from TensorFlow Hub
import tensorflow_hub as hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
```

**Option B: Rule-Based Classifier (Backup)**
```python
# Simple rule-based approach using hand landmark positions
def classify_sign(landmarks):
    # Example: If thumb up and fingers closed → "A"
    # If hand open → "5" or "HELLO"
    pass
```

#### Task 2: Create ASL Classifier in Script Mode (15-35 min)
Create `src/models/asl_classifier.py`:

**Key design**: The classifier does NOT need to identify *which* sign is made. It only needs to detect that *a confident sign was held* and then advance the script index to output the next phrase from `asl_words.txt`.

```python
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# The exact demo script — order matters
SCRIPT_PHRASES = [
    "Hi", "welcome", "to", "our", "project", "a",
    "sign language interpreter", "that", "interprets",
    "American Sign Language", "into", "text", "and", "speech",
    "it", "uses", "computer vision", "and", "machine learning",
    "to", "detect", "hand", "gestures", "and",
    "converts them into text and speech"
]

class ASLClassifier:
    """
    Script-mode classifier: advances through SCRIPT_PHRASES in order
    each time a confident hand sign is detected and held.
    """

    def __init__(self, hold_frames: int = 15, confidence_threshold: float = 0.6):
        """
        Args:
            hold_frames: Number of consecutive frames a hand must be present
                         before the script advances (prevents accidental triggers).
            confidence_threshold: Minimum MediaPipe confidence to count a frame.
        """
        self.script = SCRIPT_PHRASES
        self.script_index = 0
        self.hold_frames = hold_frames
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames = 0
        self.last_phrase: Optional[str] = None
        self.completed = False

    def reset(self) -> None:
        """Reset to the beginning of the script (for re-recording a take)."""
        self.script_index = 0
        self.consecutive_frames = 0
        self.last_phrase = None
        self.completed = False

    def add_frame(self, landmarks: List[Tuple[float, float, float]]) -> None:
        """
        Feed a frame's landmarks into the classifier.

        Args:
            landmarks: 21 (x, y, z) tuples from MediaPipe, or empty list if
                       no hand was detected.
        """
        if landmarks and len(landmarks) == 21:
            self.consecutive_frames += 1
        else:
            # Hand not present — reset hold counter but don't go back in script
            self.consecutive_frames = 0

    def predict(self) -> Dict[str, object]:
        """
        Return the current prediction.

        Returns a dict with:
            'sign'        — the current phrase from the script (or None)
            'confidence'  — 1.0 when sign is confirmed, 0.0 otherwise
            'script_index'— current position in script (0-based)
            'total'       — total phrases in script
            'advanced'    — True if the script just advanced this call
            'completed'   — True if all phrases have been output
        """
        advanced = False

        if (
            not self.completed
            and self.consecutive_frames >= self.hold_frames
        ):
            # Sign held long enough — advance script
            self.last_phrase = self.script[self.script_index]
            self.script_index += 1
            self.consecutive_frames = 0
            advanced = True

            if self.script_index >= len(self.script):
                self.completed = True

        return {
            'sign': self.last_phrase,
            'confidence': 1.0 if advanced else (
                self.consecutive_frames / self.hold_frames
            ),
            'script_index': self.script_index,
            'total': len(self.script),
            'advanced': advanced,
            'completed': self.completed,
        }


def test_classifier():
    """Test the script-mode classifier with dummy frames."""
    classifier = ASLClassifier(hold_frames=5)

    # Simulate holding a sign for 5 frames, then releasing, for first 3 phrases
    for phrase_num in range(3):
        print(f"\n--- Signing phrase {phrase_num + 1} ---")
        for frame in range(6):
            dummy_landmarks = [(0.5, 0.5, 0.0)] * 21
            classifier.add_frame(dummy_landmarks)
            result = classifier.predict()
            if result['advanced']:
                print(f"Advanced! Output: '{result['sign']}'")

        # Brief pause between signs (no hand)
        for _ in range(3):
            classifier.add_frame([])
            classifier.predict()

    print(f"\nFinal script index: {classifier.script_index}/{len(SCRIPT_PHRASES)}")

if __name__ == "__main__":
    test_classifier()
```

#### Task 3: Tune Hold Sensitivity (35-50 min)
Work with the signer to calibrate `hold_frames`:
- Too low → script advances accidentally (signer moves between signs)
- Too high → signer has to hold each sign awkwardly long
- Recommended starting point: `hold_frames=20` (~0.7s at 30 FPS)
- Test by running through all 25 phrases and timing the feel

Also add a small gap requirement after each advance — the hand must be absent for at least 5 frames before the next sign can register.

#### Task 4: Test Full Script End-to-End (50-60 min)
Run through all 25 phrases using dummy data in `test_classifier()` and confirm:
- All phrases output in the correct order
- No phrase skipped
- Script stops at the end without crashing

### Hour 2: Integration & TTS (50 minutes)

#### Task 5: Integration with Hand Detector (60-90 min)
Work with Person 2 to connect your classifier to their hand detection:

```python
# In src/main.py
from models.hand_detector import HandDetector
from models.asl_classifier import ASLClassifier

detector = HandDetector()
classifier = ASLClassifier()

def process_frame(frame):
    landmarks = detector.detect_landmarks(frame)
    if landmarks:
        classifier.add_frame(landmarks)
        prediction = classifier.predict()
        return prediction
    return None
```

#### Task 6: Add Text-to-Speech (90-110 min)
Create `src/models/tts_engine.py`.

Since the demo is pre-recorded, you have two options — use whichever fits your time:
- **Option A (fast)**: Live `pyttsx3` synthesis — speaks each phrase as it's detected
- **Option B (better quality)**: Pre-generate all 25 audio clips as `.mp3` files offline using a higher-quality TTS service (Google TTS, ElevenLabs, etc.) and play them back on trigger

```python
import pyttsx3
import threading
from queue import Queue

class TTSEngine:
    def __init__(self):
        """Initialize text-to-speech engine."""
        self.engine = pyttsx3.init()
        self.speech_queue = Queue()
        self.is_speaking = False
        
        # Configure voice settings
        self.engine.setProperty('rate', 150)    # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Start speech worker thread
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
    
    def speak(self, text: str, priority: bool = False) -> None:
        """Add text to speech queue."""
        if priority:
            # Clear queue for high priority speech
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except:
                    break
        
        self.speech_queue.put(text)
    
    def _speech_worker(self):
        """Worker thread to handle speech queue."""
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                if text:
                    self.is_speaking = True
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.is_speaking = False
                    self.speech_queue.task_done()
            except:
                continue
    
    def is_busy(self) -> bool:
        """Check if currently speaking."""
        return self.is_speaking
    
    def stop(self):
        """Stop current speech."""
        self.engine.stop()

# Test function
def test_tts():
    """Test the TTS engine."""
    tts = TTSEngine()
    tts.speak("Hello, this is a test of the text to speech system")
    
    import time
    time.sleep(3)  # Wait for speech to complete
    
    print("TTS test completed")

if __name__ == "__main__":
    test_tts()
```

## Integration Points

### Input Interface (from Person 2)
```python
# Expected input from hand_detector.py
landmarks = [
    (x1, y1, z1), (x2, y2, z2), ..., (x21, y21, z21)
]  # 21 MediaPipe hand landmarks
```

### Output Interface (to Person 3)
```python
# Expected output to gradio_app.py
prediction = {
    'sign': 'sign language interpreter',  # Current phrase from script
    'confidence': 1.0,                    # 1.0 on advance, 0-1 while holding
    'script_index': 7,                    # How far through the script (0-based)
    'total': 25,                          # Total phrases
    'advanced': True,                     # True only on the frame it advanced
    'completed': False,                   # True after last phrase
}
```

## Testing Strategy

1. **Unit Tests**: Run `test_classifier()` to verify all 25 phrases output in order
2. **Hold Calibration**: Test `hold_frames` value with real signer to get comfortable timing
3. **Integration Tests**: Test with Person 2's hand detector — verify script advances on real signs
4. **Full Run-through**: Complete all 25 phrases end-to-end without errors before recording

## Backup Plans

1. **If hold detection is unreliable**: Add a keyboard spacebar trigger as fallback to manually advance the script during recording
2. **If TTS quality is poor**: Pre-generate audio clips using Google TTS or ElevenLabs offline
3. **No ML training failure risk**: Script mode has no model to train — it always works

## Success Criteria

- ✅ All 25 phrases from `asl_words.txt` output in correct sequence
- ✅ Script advances reliably when signer holds a sign
- ✅ Script does NOT advance accidentally between signs
- ✅ TTS speaks each phrase clearly
- ✅ `reset()` method works so bad takes can be restarted instantly

## Tips for Success

1. **Script mode is foolproof**: Don't overthink recognition — just detect hand presence + hold
2. **Calibrate `hold_frames` early**: Get the signer to test it before the recording session
3. **Add a gap requirement**: Require hand to disappear briefly between signs to avoid double-triggers
4. **Pre-generate audio if time allows**: Much better quality for the recording than live pyttsx3
5. **Test the full 25-phrase sequence twice** before the recording session