# Person 2: Computer Vision Developer - Tasks & Guidelines

## Role Overview
You are responsible for the "eyes" of the ASL interpreter - capturing video input, detecting hands, extracting landmarks, and providing clean data to the ML model.

## Timeline & Tasks

### Hour 1: Hand Detection Implementation (60 minutes)

#### Task 1: MediaPipe Setup & Testing (0-20 min)
First, set up MediaPipe and test basic hand detection:

```python
# Test script to verify MediaPipe works
import cv2
import mediapipe as mp
import numpy as np

# Test MediaPipe installation
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def test_mediapipe():
    """Test MediaPipe hand detection with webcam."""
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    print("Testing MediaPipe - Press 'q' to quit")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the image
        results = hands.process(image_rgb)
        
        # Draw the hand annotations
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                print(f"Hand detected with {len(hand_landmarks.landmark)} landmarks")
        
        cv2.imshow('MediaPipe Hands Test', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("MediaPipe test completed successfully!")

if __name__ == "__main__":
    test_mediapipe()
```

#### Task 2: Create Hand Detector Class (20-45 min)
Create `src/models/hand_detector.py`:

```python
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import time

class HandDetector:
    def __init__(self, 
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe hand detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Frame processing statistics
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_time = time.time()
        
    def detect_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect hand landmarks in a frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary containing landmarks and metadata, or None if no hand detected
        """
        self.frame_count += 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for drawing
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            self.detection_count += 1
            self.last_detection_time = time.time()
            
            # Get the first (primary) hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmark coordinates
            landmarks = self._extract_coordinates(hand_landmarks, frame.shape)
            
            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS
            )
            
            return {
                'landmarks': landmarks,
                'raw_landmarks': hand_landmarks,
                'annotated_frame': annotated_frame,
                'hand_present': True,
                'confidence': self._calculate_confidence(hand_landmarks),
                'timestamp': time.time(),
                'frame_id': self.frame_count
            }
        
        return {
            'landmarks': None,
            'raw_landmarks': None,
            'annotated_frame': annotated_frame,
            'hand_present': False,
            'confidence': 0.0,
            'timestamp': time.time(),
            'frame_id': self.frame_count
        }
    
    def _extract_coordinates(self, hand_landmarks, frame_shape) -> List[Tuple[float, float, float]]:
        """
        Extract normalized coordinates from MediaPipe landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the input frame (height, width, channels)
            
        Returns:
            List of (x, y, z) coordinates for 21 hand landmarks
        """
        height, width = frame_shape[:2]
        coordinates = []
        
        for landmark in hand_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates, then back to normalized
            x = landmark.x  # Already normalized (0-1)
            y = landmark.y  # Already normalized (0-1)
            z = landmark.z  # Depth relative to wrist
            
            coordinates.append((x, y, z))
        
        return coordinates
    
    def _calculate_confidence(self, hand_landmarks) -> float:
        """
        Calculate confidence score based on landmark visibility.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence calculation based on landmark visibility
        visible_landmarks = sum(1 for lm in hand_landmarks.landmark if lm.visibility > 0.5)
        total_landmarks = len(hand_landmarks.landmark)
        
        return visible_landmarks / total_landmarks if total_landmarks > 0 else 0.0
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        detection_rate = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        time_since_last_detection = time.time() - self.last_detection_time
        
        return {
            'total_frames': self.frame_count,
            'detections': self.detection_count,
            'detection_rate': detection_rate,
            'time_since_last_detection': time_since_last_detection
        }
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_time = time.time()
    
    def close(self):
        """Clean up resources."""
        self.hands.close()

# Test function
def test_hand_detector():
    """Test the HandDetector class with webcam."""
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    print("Testing HandDetector - Press 'q' to quit, 's' for statistics")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        # Detect hands
        result = detector.detect_landmarks(frame)
        
        # Display results
        if result['hand_present']:
            print(f"Hand detected! Confidence: {result['confidence']:.2f}")
            cv2.imshow('Hand Detection', result['annotated_frame'])
        else:
            cv2.imshow('Hand Detection', result['annotated_frame'])
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            stats = detector.get_statistics()
            print(f"Statistics: {stats}")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("HandDetector test completed!")

if __name__ == "__main__":
    test_hand_detector()
```

#### Task 3: Add Frame Buffering (45-60 min)
Enhance the detector with frame buffering for temporal analysis:

```python
# Add to HandDetector class
from collections import deque

class HandDetector:
    def __init__(self, ...):
        # ... existing init code ...
        
        # Frame buffering for temporal analysis
        self.frame_buffer_size = 5
        self.landmark_buffer = deque(maxlen=self.frame_buffer_size)
        self.confidence_buffer = deque(maxlen=self.frame_buffer_size)
    
    def get_temporal_landmarks(self) -> Optional[List[List[Tuple[float, float, float]]]]:
        """
        Get buffered landmarks for temporal analysis.
        
        Returns:
            List of landmark sequences, or None if buffer not full
        """
        if len(self.landmark_buffer) < self.frame_buffer_size:
            return None
        
        return list(self.landmark_buffer)
    
    def get_smoothed_landmarks(self) -> Optional[List[Tuple[float, float, float]]]:
        """
        Get smoothed landmarks by averaging recent detections.
        
        Returns:
            Smoothed landmarks, or None if no recent detections
        """
        if not self.landmark_buffer:
            return None
        
        # Simple averaging of recent landmarks
        recent_landmarks = [lm for lm in self.landmark_buffer if lm is not None]
        
        if not recent_landmarks:
            return None
        
        # Average each landmark position
        smoothed = []
        for i in range(21):  # 21 hand landmarks
            x_sum = sum(landmarks[i][0] for landmarks in recent_landmarks)
            y_sum = sum(landmarks[i][1] for landmarks in recent_landmarks)
            z_sum = sum(landmarks[i][2] for landmarks in recent_landmarks)
            
            count = len(recent_landmarks)
            smoothed.append((x_sum/count, y_sum/count, z_sum/count))
        
        return smoothed
    
    # Update detect_landmarks method to use buffering
    def detect_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        # ... existing detection code ...
        
        if results.multi_hand_landmarks:
            # ... existing processing ...
            
            # Add to buffers
            self.landmark_buffer.append(landmarks)
            self.confidence_buffer.append(confidence)
            
            # Add smoothed landmarks to result
            result['smoothed_landmarks'] = self.get_smoothed_landmarks()
            result['temporal_landmarks'] = self.get_temporal_landmarks()
            
        else:
            # No hand detected
            self.landmark_buffer.append(None)
            self.confidence_buffer.append(0.0)
            
            result['smoothed_landmarks'] = None
            result['temporal_landmarks'] = None
        
        return result
```

### Hour 2: Integration & Optimization (50 minutes)

#### Task 4: Integration with ML Model (60-90 min)
Work with Person 1 to connect your detector to their classifier:

```python
# Integration example in src/main.py
from models.hand_detector import HandDetector
from models.asl_classifier import ASLClassifier

def create_cv_ml_pipeline():
    """Create integrated CV + ML pipeline."""
    detector = HandDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    classifier = ASLClassifier()
    
    def process_frame(frame):
        # Detect hands
        detection_result = detector.detect_landmarks(frame)
        
        if detection_result['hand_present']:
            # Add landmarks to classifier
            classifier.add_frame(detection_result['landmarks'])
            
            # Get prediction
            prediction = classifier.predict()
            
            return {
                'frame': detection_result['annotated_frame'],
                'prediction': prediction,
                'confidence': detection_result['confidence'],
                'landmarks_detected': True
            }
        else:
            return {
                'frame': detection_result['annotated_frame'],
                'prediction': None,
                'confidence': 0.0,
                'landmarks_detected': False
            }
    
    return process_frame, detector, classifier
```

#### Task 5: Reliability Over Speed (90-110 min)
Since the demo is pre-recorded (not live), focus on **detection reliability** rather than frame rate optimisation:

- Raise `min_detection_confidence` to `0.7` for cleaner landmark data
- Ensure the annotated frame (with hand tracking overlay) looks good visually — it will be on screen during the recording
- Test under the actual lighting conditions you'll use for recording
- No frame-skipping needed — process every frame for best accuracy

## Integration Points

### Output Interface (to Person 1)
```python
# Expected output format for asl_classifier.py
landmarks = [
    (x1, y1, z1), (x2, y2, z2), ..., (x21, y21, z21)
]  # 21 MediaPipe hand landmarks, normalized coordinates (0-1)
```

### Input Interface (from Person 3)
```python
# Expected input from gradio_app.py
frame = np.ndarray  # BGR image from webcam, shape (height, width, 3)
```

## Testing Strategy

1. **Unit Tests**: Test landmark extraction with known images
2. **Performance Tests**: Measure FPS and processing time
3. **Integration Tests**: Test with Person 1's classifier
4. **Edge Cases**: Test with poor lighting, multiple hands, no hands

## Backup Plans

1. **If MediaPipe fails**: Use OpenCV contour detection
2. **If performance is slow**: Reduce frame rate or resolution
3. **If accuracy is poor**: Adjust confidence thresholds
4. **If integration fails**: Provide mock landmark data

## Success Criteria

- ✅ MediaPipe detects hands reliably under recording lighting conditions
- ✅ Extracts 21 hand landmarks accurately
- ✅ Hand tracking overlay looks clean and professional on screen
- ✅ `hand_present: True` fires consistently when the signer holds a sign
- ✅ `hand_present: False` fires cleanly between signs (no ghost detections)
- ✅ Integration works smoothly with Person 1's script-mode classifier

## Tips for Success

1. **Test Early**: Verify MediaPipe works with your webcam first
2. **Start Simple**: Basic detection before adding optimizations
3. **Monitor Performance**: Track FPS and processing time
4. **Handle Edge Cases**: No hands, multiple hands, poor lighting
5. **Coordinate with Person 1**: Ensure data format compatibility
6. **Use Visualization**: Draw landmarks for debugging