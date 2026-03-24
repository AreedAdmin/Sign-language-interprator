# Person 2 — Computer Vision Developer: Agent Instructions

## Your Role

You are building the **hand detection component** — the "eyes" of the ASL
interpreter. You capture webcam frames, detect hands via MediaPipe, extract
landmarks, and feed clean data to Person 1's classifier.

---

## Critical Context

- The demo is a **pre-recorded video**, not a live show. **Reliability and
  visual quality matter more than processing speed.**
- You do **not** need to identify *which* ASL sign is made. Just detect that a
  hand is present and return landmarks.
- Person 1's classifier handles script advancement based on consecutive frames
  of hand presence.

---

## Task 1 — Create `src/models/hand_detector.py`

Implement `class HandDetector` using MediaPipe Hands.

### `detect_landmarks(frame)` return format

```python
{
    'landmarks':       List[Tuple[float,float,float]],  # 21 (x,y,z) normalised tuples
    'annotated_frame': np.ndarray,   # Frame with landmark overlay drawn on it
    'hand_present':    bool,         # True if hand detected with sufficient confidence
    'confidence':      float,        # 0.0 – 1.0
    'timestamp':       float,        # time.time()
}
```

### Key configuration

```python
self.hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,   # Higher than default for cleaner output
    min_tracking_confidence=0.5,
)
```

### Drawing the overlay

Use `mp_drawing.draw_landmarks()` to draw the hand skeleton on the frame.
**The annotated frame will be visible in the screen recording — make it look
clean.** Use the default drawing styles or customise colours if time allows.

### No frame-skipping

Do **not** add frame-skipping optimisation. Process every frame for maximum
reliability. The demo is pre-recorded, so throughput is not a concern.

### Skeleton

```python
import cv2
import mediapipe as mp
import time
from typing import List, Tuple, Optional, Dict
import numpy as np

class HandDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_landmarks(self, frame: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                annotated, hand, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            return {
                'landmarks': landmarks,
                'annotated_frame': annotated,
                'hand_present': True,
                'confidence': sum(lm.visibility for lm in hand.landmark) / 21,
                'timestamp': time.time(),
            }

        return {
            'landmarks': [],
            'annotated_frame': annotated,
            'hand_present': False,
            'confidence': 0.0,
            'timestamp': time.time(),
        }

    def close(self):
        self.hands.close()
```

---

## Task 2 — Test Under Recording Conditions

Before the recording session:

1. Run the detector with your webcam under the **exact lighting conditions**
   you'll use for the recording.
2. Confirm `hand_present=True` fires consistently when the signer holds a sign.
3. Confirm `hand_present=False` fires cleanly when the hand is lowered between
   signs (no "ghost" detections).
4. Verify the annotated frame looks clean on screen.

---

## Task 3 — Support Integration

Work with Person 1 to confirm the landmark format is compatible:

- 21 tuples of `(x, y, z)` where x, y are normalised 0–1 values.
- Landmarks in MediaPipe order (wrist = index 0).

---

## Integration Interface

### Output to Person 1

```python
# detect_landmarks() output — directly consumed by ASLClassifier.add_frame()
result['landmarks']       # List of 21 (x,y,z) tuples — pass to classifier
result['annotated_frame'] # Pass to Gradio UI for display
result['hand_present']    # True/False — drives consecutive frame counting
```

---

## Success Criteria

- MediaPipe detects hands reliably under recording lighting conditions
- Hand tracking overlay looks clean and professional on screen
- `hand_present: True` fires consistently when the signer holds a sign
- `hand_present: False` fires cleanly between signs (no ghost detections)
- Smooth integration with Person 1's script-mode classifier

---

## Tips

1. Test early with your actual webcam and lighting — MediaPipe behaves
   differently under different conditions.
2. If the overlay looks noisy, increase `min_detection_confidence` to `0.8`.
3. The annotated frame is on screen during the recording — it represents the
   project visually. Keep it clean.
4. Coordinate with Person 1 on landmark format before integration.
