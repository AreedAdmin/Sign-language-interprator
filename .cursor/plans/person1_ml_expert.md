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

#### Task 2: Create ASL Classifier (15-35 min)
Create `src/models/asl_classifier.py`:

```python
import numpy as np
from typing import List, Dict, Tuple

class ASLClassifier:
    def __init__(self):
        """Initialize the ASL classifier with basic signs."""
        self.signs = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
            5: 'HELLO', 6: 'THANK_YOU', 7: 'YES', 8: 'NO', 9: 'PLEASE'
        }
        self.model = self._load_model()
        self.frame_buffer = []
        self.buffer_size = 5
    
    def _load_model(self):
        """Load or create the classification model."""
        # TODO: Implement model loading
        # For now, return None and use rule-based classification
        return None
    
    def add_frame(self, landmarks: List[Tuple[float, float, float]]) -> None:
        """Add a frame of landmarks to the buffer."""
        if len(landmarks) == 21:  # MediaPipe hands gives 21 landmarks
            normalized = self._normalize_landmarks(landmarks)
            self.frame_buffer.append(normalized)
            
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
    
    def _normalize_landmarks(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """Normalize landmarks relative to wrist position."""
        if not landmarks:
            return [0.0] * 63  # 21 points * 3 coordinates
        
        # Use wrist (landmark 0) as reference point
        wrist = landmarks[0]
        normalized = []
        
        for point in landmarks:
            normalized.extend([
                point[0] - wrist[0],  # x relative to wrist
                point[1] - wrist[1],  # y relative to wrist
                point[2] - wrist[2]   # z relative to wrist
            ])
        
        return normalized
    
    def predict(self) -> Dict[str, any]:
        """Predict the current sign based on frame buffer."""
        if len(self.frame_buffer) < 3:
            return {'sign': None, 'confidence': 0.0}
        
        if self.model is None:
            # Use rule-based classification
            return self._rule_based_prediction()
        else:
            # Use ML model
            return self._ml_prediction()
    
    def _rule_based_prediction(self) -> Dict[str, any]:
        """Simple rule-based sign recognition."""
        if not self.frame_buffer:
            return {'sign': None, 'confidence': 0.0}
        
        latest_frame = self.frame_buffer[-1]
        
        # Simple rules based on hand shape
        # These are very basic - you can improve them
        
        # Check if hand is in fist position (A)
        if self._is_fist(latest_frame):
            return {'sign': 'A', 'confidence': 0.8}
        
        # Check if hand is open (HELLO or 5)
        elif self._is_open_hand(latest_frame):
            return {'sign': 'HELLO', 'confidence': 0.7}
        
        # Check if thumb up (YES)
        elif self._is_thumbs_up(latest_frame):
            return {'sign': 'YES', 'confidence': 0.75}
        
        # Default
        return {'sign': 'UNKNOWN', 'confidence': 0.3}
    
    def _is_fist(self, landmarks: List[float]) -> bool:
        """Check if hand is in fist position."""
        # Simple check: fingers are close to palm
        # This is a placeholder - implement based on landmark positions
        return False
    
    def _is_open_hand(self, landmarks: List[float]) -> bool:
        """Check if hand is open."""
        # Simple check: fingers are extended
        # This is a placeholder - implement based on landmark positions
        return True
    
    def _is_thumbs_up(self, landmarks: List[float]) -> bool:
        """Check if thumb is up."""
        # Simple check: thumb is extended upward
        # This is a placeholder - implement based on landmark positions
        return False
    
    def _ml_prediction(self) -> Dict[str, any]:
        """Use ML model for prediction."""
        # TODO: Implement ML model prediction
        # Convert frame_buffer to model input format
        # Run inference
        # Return prediction with confidence
        pass

# Test function
def test_classifier():
    """Test the classifier with dummy data."""
    classifier = ASLClassifier()
    
    # Dummy landmarks (21 points with x, y, z coordinates)
    dummy_landmarks = [(0.5, 0.5, 0.0) for _ in range(21)]
    
    classifier.add_frame(dummy_landmarks)
    result = classifier.predict()
    
    print(f"Test prediction: {result}")
    return result

if __name__ == "__main__":
    test_classifier()
```

#### Task 3: Implement Basic Sign Recognition (35-50 min)
Focus on these 7-10 signs:
- **Letters**: A, B, C, D, E
- **Words**: HELLO, THANK_YOU, YES, NO, PLEASE

Implement the rule-based methods:
- `_is_fist()` - for letter "A"
- `_is_open_hand()` - for "HELLO" 
- `_is_thumbs_up()` - for "YES"
- Add 2-3 more simple gestures

#### Task 4: Test with Dummy Data (50-60 min)
Create test cases and verify your classifier works with mock landmark data.

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
Create `src/models/tts_engine.py`:

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
    'sign': 'HELLO',
    'confidence': 0.87,
    'alternatives': [('HELLO', 0.87), ('HI', 0.10), ('WAVE', 0.03)]
}
```

## Testing Strategy

1. **Unit Tests**: Test each method with known inputs
2. **Mock Data**: Use dummy landmarks to test classification
3. **Integration Tests**: Test with Person 2's hand detector
4. **Performance Tests**: Ensure predictions are fast (<100ms)

## Backup Plans

1. **If ML model fails**: Use rule-based classification
2. **If rules are complex**: Start with just 3-4 basic signs
3. **If TTS fails**: Use simple print statements for demo

## Success Criteria

- ✅ ASL classifier processes hand landmarks
- ✅ Recognizes at least 7 basic signs
- ✅ Returns predictions with confidence scores
- ✅ TTS speaks detected signs clearly
- ✅ Integration works smoothly with other components
- ✅ Processing time under 100ms per prediction

## Tips for Success

1. **Start Simple**: Begin with 3-4 obvious signs (A, HELLO, YES)
2. **Test Early**: Verify each component works independently
3. **Use Placeholders**: Mock complex parts initially
4. **Focus on Integration**: Spend time ensuring smooth data flow
5. **Have Backups**: Prepare fallback solutions for each component