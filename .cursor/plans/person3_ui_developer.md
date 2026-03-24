# Person 3: UI/Frontend Developer - Tasks & Guidelines

## Role Overview
You are responsible for creating the user interface that brings everything together - the Gradio web application that displays video, shows detected signs, and provides audio controls.

## Timeline & Tasks

### Hour 1: Gradio Interface Development (60 minutes)

#### Task 1: Basic Gradio Setup & Testing (0-25 min)
First, set up and test basic Gradio functionality:

```python
# Test script to verify Gradio works
import gradio as gr
import numpy as np
import cv2
import time

def test_gradio_basic():
    """Test basic Gradio functionality."""
    
    def simple_video_process(frame):
        """Simple video processing function."""
        if frame is None:
            return frame, "No video input"
        
        # Add timestamp overlay
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, f"Processing frame at {timestamp}"
    
    # Create interface
    interface = gr.Interface(
        fn=simple_video_process,
        inputs=gr.Image(source="webcam", streaming=True),
        outputs=[
            gr.Image(label="Processed Video"),
            gr.Text(label="Status")
        ],
        title="Gradio Video Test",
        description="Testing basic video processing with Gradio"
    )
    
    return interface

if __name__ == "__main__":
    demo = test_gradio_basic()
    demo.launch(debug=True)
```

#### Task 2: Create Main Gradio Application (25-45 min)
Create `src/ui/gradio_app.py`:

```python
import gradio as gr
import numpy as np
import cv2
import time
from typing import Optional, Dict, Any, Tuple
import threading
import queue

class ASLGradioApp:
    def __init__(self):
        """Initialize the ASL Gradio application."""
        self.current_sign = "None"
        self.sign_history = []
        self.confidence_score = 0.0
        self.detection_status = "Waiting for camera..."
        self.audio_enabled = True
        self.max_history = 10
        
        # Processing pipeline (will be set by integration)
        self.process_pipeline = None
        
        # Statistics
        self.total_detections = 0
        self.session_start_time = time.time()
        
    def set_pipeline(self, pipeline_func):
        """Set the processing pipeline function."""
        self.process_pipeline = pipeline_func
    
    def process_video_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, str, str, str]:
        """
        Process a single video frame through the ASL pipeline.
        
        Args:
            frame: Input video frame from webcam
            
        Returns:
            Tuple of (processed_frame, current_sign, confidence, status, history)
        """
        if frame is None:
            return None, self.current_sign, f"{self.confidence_score:.1%}", self.detection_status, self._format_history()
        
        try:
            # Process through pipeline (Person 4 will integrate this)
            if self.process_pipeline:
                result = self.process_pipeline(frame)
                
                if result and result.get('prediction'):
                    prediction = result['prediction']
                    
                    # Update current sign if confidence is high enough
                    if prediction.get('confidence', 0) > 0.6:
                        new_sign = prediction.get('sign', 'Unknown')
                        
                        # Only update if it's a new sign
                        if new_sign != self.current_sign:
                            self.current_sign = new_sign
                            self.confidence_score = prediction.get('confidence', 0)
                            self._add_to_history(new_sign)
                            self.total_detections += 1
                            
                            # Trigger TTS if enabled
                            if self.audio_enabled and hasattr(self, 'tts_engine'):
                                self.tts_engine.speak(new_sign)
                    
                    self.detection_status = "Detecting signs..."
                    processed_frame = result.get('frame', frame)
                    
                else:
                    self.detection_status = "No hand detected"
                    processed_frame = frame
            else:
                # No pipeline set - just pass through frame
                self.detection_status = "Pipeline not connected"
                processed_frame = self._add_overlay(frame, "No Pipeline")
            
        except Exception as e:
            self.detection_status = f"Error: {str(e)}"
            processed_frame = self._add_overlay(frame, "Error")
        
        return (
            processed_frame,
            self.current_sign,
            f"{self.confidence_score:.1%}",
            self.detection_status,
            self._format_history()
        )
    
    def _add_overlay(self, frame: np.ndarray, message: str) -> np.ndarray:
        """Add text overlay to frame."""
        if frame is None:
            return frame
        
        overlay_frame = frame.copy()
        
        # Add status message
        cv2.putText(overlay_frame, message, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(overlay_frame, f"Time: {timestamp}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def _add_to_history(self, sign: str):
        """Add sign to history."""
        self.sign_history.append({
            'sign': sign,
            'timestamp': time.strftime("%H:%M:%S"),
            'confidence': self.confidence_score
        })
        
        # Keep only recent history
        if len(self.sign_history) > self.max_history:
            self.sign_history.pop(0)
    
    def _format_history(self) -> str:
        """Format sign history for display."""
        if not self.sign_history:
            return "No signs detected yet"
        
        history_lines = []
        for entry in reversed(self.sign_history[-5:]):  # Show last 5
            history_lines.append(
                f"{entry['timestamp']}: {entry['sign']} ({entry['confidence']:.1%})"
            )
        
        return "\n".join(history_lines)
    
    def toggle_audio(self) -> str:
        """Toggle audio on/off."""
        self.audio_enabled = not self.audio_enabled
        return f"Audio: {'ON' if self.audio_enabled else 'OFF'}"
    
    def clear_history(self) -> Tuple[str, str]:
        """Clear sign history."""
        self.sign_history = []
        self.current_sign = "None"
        self.confidence_score = 0.0
        self.total_detections = 0
        return "None", "No signs detected yet"
    
    def get_statistics(self) -> str:
        """Get session statistics."""
        session_duration = time.time() - self.session_start_time
        detection_rate = self.total_detections / (session_duration / 60) if session_duration > 0 else 0
        
        stats = f"""
        Session Duration: {session_duration/60:.1f} minutes
        Total Detections: {self.total_detections}
        Detection Rate: {detection_rate:.1f} signs/minute
        Audio Enabled: {self.audio_enabled}
        """
        
        return stats.strip()
    
    def create_interface(self) -> gr.Interface:
        """Create the main Gradio interface."""
        
        with gr.Blocks(title="ASL Real-time Interpreter", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🤟 ASL Real-time Interpreter")
            gr.Markdown("Point your camera at ASL signs to see real-time detection and hear audio output.")
            
            with gr.Row():
                # Left column - Video
                with gr.Column(scale=2):
                    video_input = gr.Image(
                        source="webcam", 
                        streaming=True, 
                        label="Camera Feed",
                        type="numpy"
                    )
                    
                    video_output = gr.Image(
                        label="Processed Video with Hand Tracking",
                        type="numpy"
                    )
                
                # Right column - Detection results
                with gr.Column(scale=1):
                    gr.Markdown("## Detection Results")
                    
                    current_sign_display = gr.Textbox(
                        label="Current Sign",
                        value="None",
                        interactive=False
                    )
                    
                    confidence_display = gr.Textbox(
                        label="Confidence",
                        value="0%",
                        interactive=False
                    )
                    
                    status_display = gr.Textbox(
                        label="Status",
                        value="Waiting for camera...",
                        interactive=False
                    )
                    
                    gr.Markdown("## Recent Signs")
                    history_display = gr.Textbox(
                        label="Sign History",
                        value="No signs detected yet",
                        lines=5,
                        interactive=False
                    )
                    
                    # Controls
                    gr.Markdown("## Controls")
                    with gr.Row():
                        audio_toggle_btn = gr.Button("🔊 Toggle Audio")
                        clear_history_btn = gr.Button("🗑️ Clear History")
                    
                    audio_status = gr.Textbox(
                        label="Audio Status",
                        value="Audio: ON",
                        interactive=False
                    )
                    
                    # Statistics
                    gr.Markdown("## Statistics")
                    stats_display = gr.Textbox(
                        label="Session Stats",
                        lines=4,
                        interactive=False
                    )
                    
                    stats_refresh_btn = gr.Button("📊 Refresh Stats")
            
            # Set up event handlers
            video_input.stream(
                fn=self.process_video_frame,
                inputs=[video_input],
                outputs=[
                    video_output,
                    current_sign_display,
                    confidence_display,
                    status_display,
                    history_display
                ],
                stream_every=0.1  # Process 10 times per second
            )
            
            audio_toggle_btn.click(
                fn=self.toggle_audio,
                outputs=[audio_status]
            )
            
            clear_history_btn.click(
                fn=self.clear_history,
                outputs=[current_sign_display, history_display]
            )
            
            stats_refresh_btn.click(
                fn=self.get_statistics,
                outputs=[stats_display]
            )
        
        return demo

# Test function
def test_gradio_app():
    """Test the ASL Gradio application."""
    
    # Create mock processing pipeline simulating script-mode advancement
    script = [
        "Hi", "welcome", "to", "our", "project", "a",
        "sign language interpreter", "that", "interprets",
        "American Sign Language", "into", "text", "and", "speech",
        "it", "uses", "computer vision", "and", "machine learning",
        "to", "detect", "hand", "gestures", "and",
        "converts them into text and speech"
    ]
    mock_index = [0]  # mutable for closure

    def mock_pipeline(frame):
        """Mock pipeline: advances through script on every 30th call."""
        import random
        mock_pipeline.call_count = getattr(mock_pipeline, 'call_count', 0) + 1

        if mock_pipeline.call_count % 30 == 0 and mock_index[0] < len(script):
            phrase = script[mock_index[0]]
            mock_index[0] += 1
            return {
                'frame': frame,
                'prediction': {
                    'sign': phrase,
                    'confidence': 1.0,
                    'script_index': mock_index[0],
                    'total': len(script),
                    'advanced': True,
                    'completed': mock_index[0] >= len(script),
                }
            }
        return {'frame': frame, 'prediction': None}
    
    # Create and test app
    app = ASLGradioApp()
    app.set_pipeline(mock_pipeline)
    
    interface = app.create_interface()
    interface.launch(debug=True, share=False)

if __name__ == "__main__":
    test_gradio_app()
```

#### Task 3: Add Styling and Layout Improvements (45-60 min)
Enhance the interface with better styling:

```python
# Add to ASLGradioApp class
def create_interface(self) -> gr.Interface:
    """Create the main Gradio interface with enhanced styling."""
    
    # Custom CSS for better appearance
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    
    .sign-display {
        font-size: 24px !important;
        font-weight: bold !important;
        text-align: center !important;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    .confidence-display {
        font-size: 18px !important;
        text-align: center !important;
    }
    
    .history-box {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        font-family: monospace !important;
    }
    
    .control-button {
        background: linear-gradient(45deg, #2196F3, #21CBF3) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
    }
    
    .stats-box {
        background-color: #e3f2fd !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    """
    
    with gr.Blocks(
        title="ASL Real-time Interpreter", 
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        
        # Header with logo/title
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #667eea; margin-bottom: 10px;">🤟 ASL Real-time Interpreter</h1>
            <p style="color: #666; font-size: 16px;">
                Point your camera at ASL signs for real-time detection and audio output
            </p>
        </div>
        """)
        
        # Rest of interface code with CSS classes applied...
        # (Apply elem_classes to components for styling)
```

### Hour 2: Integration & Polish (50 minutes)

#### Task 4: Integration with Backend Pipeline (60-90 min)
Work with Person 4 to connect your UI to the processing pipeline:

```python
# Integration interface for main.py
def create_integrated_app(hand_detector, asl_classifier, tts_engine):
    """Create integrated Gradio app with all components."""
    
    app = ASLGradioApp()
    
    # Set TTS engine
    app.tts_engine = tts_engine
    
    # Create processing pipeline
    def integrated_pipeline(frame):
        """Integrated processing pipeline."""
        try:
            # Step 1: Detect hands (Person 2's component)
            detection_result = hand_detector.detect_landmarks(frame)
            
            if detection_result['hand_present']:
                # Step 2: Classify sign (Person 1's component)
                asl_classifier.add_frame(detection_result['landmarks'])
                prediction = asl_classifier.predict()
                
                return {
                    'frame': detection_result['annotated_frame'],
                    'prediction': prediction,
                    'landmarks_detected': True
                }
            else:
                return {
                    'frame': detection_result['annotated_frame'],
                    'prediction': None,
                    'landmarks_detected': False
                }
        except Exception as e:
            print(f"Pipeline error: {e}")
            return {
                'frame': frame,
                'prediction': None,
                'landmarks_detected': False
            }
    
    # Set pipeline
    app.set_pipeline(integrated_pipeline)
    
    return app.create_interface()
```

#### Task 5: UI Polish & Demo Preparation (90-110 min)
Add final touches for the demo:

```python
# Add demo-specific features
class ASLGradioApp:
    def __init__(self):
        # ... existing init ...
        
        # Script phrases for display reference
        self.script_phrases = [
            "Hi", "welcome", "to", "our", "project", "a",
            "sign language interpreter", "that", "interprets",
            "American Sign Language", "into", "text", "and", "speech",
            "it", "uses", "computer vision", "and", "machine learning",
            "to", "detect", "hand", "gestures", "and",
            "converts them into text and speech"
        ]
        
    def enable_demo_mode(self):
        """Enable demo mode with helpful hints."""
        self.demo_mode = True
    
    def create_demo_interface(self) -> gr.Interface:
        """Create demo-optimized interface."""
        
        with gr.Blocks(title="ASL Interpreter Demo") as demo:
            gr.HTML("""
            <div style="text-align: center; background: linear-gradient(45deg, #667eea, #764ba2); 
                        color: white; padding: 30px; border-radius: 15px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.5em;">🤟 ASL Real-time Interpreter</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
                    Live Sign Language Detection & Voice Synthesis
                </p>
            </div>
            """)
            
            # Demo instructions
            with gr.Accordion("📋 Demo Script", open=True):
                gr.Markdown("""
                ### Scripted Presentation Sequence:
                The signer performs 25 signs in order. Each sign advances the script.

                > *"Hi, welcome to our project — a sign language interpreter that
                > interprets American Sign Language into text and speech. It uses
                > computer vision and machine learning to detect hand gestures and
                > converts them into text and speech."*

                ### Tips for Recording:
                - Hold each sign steady for ~1 second until you hear/see it trigger
                - Lower your hand briefly between each sign
                - Use the Reset button to restart if a take goes wrong
                - Good lighting + clear hand visibility = reliable detection
                """)
            
            # Main interface (same as before but with demo styling)
            # ... rest of interface code ...
        
        return demo
```

## Integration Points

### Input Interface (from Person 4)
```python
# Expected input format from main.py integration
pipeline_function = callable  # Function that processes video frames
tts_engine = object          # Text-to-speech engine object
```

### Output Interface (to Users)
```python
# Gradio interface provides:
- Live video feed with hand tracking overlay
- Real-time sign detection display
- Confidence scores
- Sign history log
- Audio controls
- Session statistics
```

## Testing Strategy

1. **UI Tests**: Test all interface components independently
2. **Integration Tests**: Test with mock pipeline data
3. **User Experience Tests**: Test with real users
4. **Performance Tests**: Ensure smooth video streaming

## Backup Plans

1. **If Gradio fails**: Use Streamlit or Flask alternative
2. **If video streaming is slow**: Reduce frame rate
3. **If styling breaks**: Use default Gradio theme
4. **If integration fails**: Show mock data for demo

## Success Criteria

- ✅ Gradio interface loads and displays video feed with hand tracking overlay
- ✅ Current phrase from the script displays prominently when a sign is detected
- ✅ Script progress indicator shows (e.g. "Phrase 7 of 25")
- ✅ TTS audio plays for each phrase on detection
- ✅ Reset button works to restart the script sequence for a new take
- ✅ Clean, professional appearance — the UI will be visible in the recording

## Tips for Success

1. **Test with mock pipeline first**: Verify the UI advances through all 25 phrases before integrating real components
2. **Make current phrase very visible**: Large font, high contrast — this is what the audience will read in the recording
3. **Add a Reset button**: Essential for re-recording takes without restarting the app
4. **Keep it clean**: The UI is part of the demo video — avoid cluttered layouts
5. **Coordinate with Person 4**: Ensure the `script_index` and `total` fields from the prediction are displayed