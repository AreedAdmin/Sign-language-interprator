# Person 1 — ML Expert: Agent Instructions

## Your Role

You are building the **ASL classifier** and **text-to-speech engine** for a
pre-recorded hackathon demo.

---

## Critical Context

- The demo is a **pre-recorded video** (not live). Re-recording is allowed if a
  take goes wrong.
- The classifier works in **Script Mode**: it advances through `asl_words.txt`
  in order each time a confident sign is held. There is **no free-form sign
  recognition** needed.
- `asl_words.txt` contains **25 phrases** in exact order. That is your full
  vocabulary.

---

## Task 1 — Create `src/models/asl_classifier.py`

Implement `class ASLClassifier` with the following behaviour:

- Load `SCRIPT_PHRASES` from `asl_words.txt` at the top of the file (25 lines).
- Accept landmark frames from Person 2's `HandDetector` via `add_frame(landmarks)`.
- Count **consecutive frames** where a hand is present.
- When hand is held for `hold_frames` (default `20`) consecutive frames:
  - Advance `script_index` by 1.
  - Output the next phrase from the script.
  - Reset the consecutive frame counter to 0.
- Require the hand to **disappear for ≥ 5 frames** before the next sign can
  register (prevents double-triggers when transitioning between signs).
- Expose a **`reset()` method** to restart the script from phrase 1 — essential
  for re-recording bad takes without restarting the app.

### `predict()` return format

```python
{
    'sign':         str,    # Current phrase from script, or None
    'confidence':   float,  # 0.0–1.0 while holding; 1.0 on the advance frame
    'script_index': int,    # How far through the script (0-based, after advance)
    'total':        int,    # Total phrases = 25
    'advanced':     bool,   # True ONLY on the frame the script advances
    'completed':    bool,   # True after the last phrase is output
}
```

### Quick skeleton

```python
SCRIPT_PHRASES = [line.strip() for line in open('asl_words.txt') if line.strip()]

class ASLClassifier:
    def __init__(self, hold_frames=20, gap_frames=5):
        self.script = SCRIPT_PHRASES
        self.script_index = 0
        self.hold_frames = hold_frames
        self.gap_frames = gap_frames
        self.consecutive = 0
        self.gap_counter = 0
        self.last_phrase = None
        self.completed = False

    def reset(self):
        self.script_index = 0
        self.consecutive = 0
        self.gap_counter = 0
        self.last_phrase = None
        self.completed = False

    def add_frame(self, landmarks):
        """Call with 21-point landmark list (or empty list if no hand)."""
        if landmarks and len(landmarks) == 21:
            if self.gap_counter >= self.gap_frames:
                self.consecutive += 1
            self.gap_counter = 0
        else:
            self.gap_counter += 1
            self.consecutive = 0

    def predict(self):
        advanced = False
        if (not self.completed
                and self.consecutive >= self.hold_frames
                and self.script_index < len(self.script)):
            self.last_phrase = self.script[self.script_index]
            self.script_index += 1
            self.consecutive = 0
            advanced = True
            if self.script_index >= len(self.script):
                self.completed = True
        return {
            'sign': self.last_phrase,
            'confidence': min(self.consecutive / self.hold_frames, 1.0),
            'script_index': self.script_index,
            'total': len(self.script),
            'advanced': advanced,
            'completed': self.completed,
        }
```

---

## Task 2 — Create `src/models/tts_engine.py`

Implement `class TTSEngine` using `pyttsx3`.

- `speak(text)` — queues text for speech in a background thread (non-blocking).
- `stop()` — stops current speech.

**Optional upgrade** (better audio quality for recording):
Pre-generate all 25 phrases as `.mp3` files offline using Google TTS or
ElevenLabs, then play them back on trigger instead of live synthesis.

```python
import pyttsx3, threading
from queue import Queue

class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        self.queue = Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def speak(self, text):
        self.queue.put(text)

    def _worker(self):
        while True:
            text = self.queue.get(timeout=None)
            if text:
                self.engine.say(text)
                self.engine.runAndWait()

    def stop(self):
        self.engine.stop()
```

---

## Task 3 — Test the Full Script End-to-End

Write a `test_classifier()` function that simulates a full 25-phrase run with
dummy landmark data and confirms:

- All 25 phrases output in the correct order.
- No phrase is skipped.
- Script stops (does not crash) after the last phrase.
- `reset()` correctly restarts from phrase 1.

---

## Integration Interface

### Input (from Person 2)

```python
landmarks = [(x1,y1,z1), (x2,y2,z2), ...]  # 21 tuples, or [] if no hand
```

### Output (to Person 3)

```python
{
    'sign':         'sign language interpreter',
    'confidence':   1.0,
    'script_index': 7,
    'total':        25,
    'advanced':     True,
    'completed':    False,
}
```

---

## Success Criteria

- All 25 phrases output in correct sequence
- Script advances reliably when signer holds a sign
- Script does NOT advance accidentally between signs
- TTS speaks each phrase clearly
- `reset()` restarts cleanly for new recording takes

---

## Tips

1. Script mode is simple — don't over-engineer. Confidence = hand present frame
   count ÷ hold threshold.
2. Calibrate `hold_frames` with the actual signer before recording.
3. Pre-generate audio clips if time allows — much cleaner for the final video.
4. Test the full 25-phrase sequence at least twice before the recording session.
