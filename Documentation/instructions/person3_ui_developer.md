# Person 3 — UI Developer: Agent Instructions

## Your Role

You are building the **Gradio web interface** — the visual face of the demo.
The UI will be on screen during the screen recording, so it must look clean
and professional.

---

## Critical Context

- The demo is a **pre-recorded video**. The Gradio UI is what the audience
  sees in the recording. Visual quality matters.
- The classifier outputs one of **25 phrases** from `asl_words.txt` in
  sequence — not free-form recognition.
- The `prediction` dict from Person 1 has these fields:

```python
{
    'sign':         str,    # Current phrase (e.g. "sign language interpreter")
    'confidence':   float,  # 0.0–1.0 while holding; 1.0 on advance
    'script_index': int,    # Position in script after advance (1–25)
    'total':        int,    # Always 25
    'advanced':     bool,   # True only on the frame it advances
    'completed':    bool,   # True after last phrase
}
```

---

## Task 1 — Create `src/ui/gradio_app.py`

Implement `class ASLGradioApp` with `create_interface() -> gr.Blocks`.

### Required UI elements

| Element | Purpose |
|---------|---------|
| Live webcam feed (left panel) | Shows signer with hand tracking overlay |
| **Current phrase display** (large, high contrast) | The phrase the audience reads |
| Progress indicator | "Phrase 7 of 25" |
| Confidence bar / text | Shows how close the sign is to triggering |
| **Reset button** | Restarts the script — essential for re-recording bad takes |
| TTS audio triggers automatically | Speaks each phrase on `advanced=True` |

### Current phrase display

This is the most important visual element. Make it large and impossible to miss:

```python
current_phrase = gr.Textbox(
    label="Detected Sign",
    value="—",
    lines=2,
    elem_classes=["phrase-display"],   # large font via CSS
    interactive=False,
)
```

Add custom CSS:

```python
css = """
.phrase-display textarea {
    font-size: 2rem !important;
    font-weight: bold !important;
    text-align: center !important;
    color: #1a1a2e !important;
    background: #e8f4f8 !important;
    border-radius: 12px !important;
}
"""
```

### Reset button

Wire the Reset button to call `classifier.reset()` through the pipeline:

```python
reset_btn = gr.Button("↺ Reset Script", variant="secondary")
reset_btn.click(fn=self.reset_script, outputs=[current_phrase, progress_display])
```

### `process_video_frame()` logic

```python
def process_video_frame(self, frame):
    if frame is None:
        return self._no_change()

    result = self.pipeline(frame)
    prediction = result.get('prediction') if result else None

    if prediction and prediction.get('advanced'):
        phrase = prediction['sign']
        self.current_phrase = phrase
        self.tts_engine.speak(phrase)

    progress = (
        f"Phrase {prediction['script_index']} of {prediction['total']}"
        if prediction else "Waiting..."
    )
    confidence_pct = f"{prediction['confidence']:.0%}" if prediction else "0%"

    return (
        result['frame'] if result else frame,
        self.current_phrase or "—",
        progress,
        confidence_pct,
    )
```

---

## Task 2 — Mock Pipeline for Independent Testing

Use this mock to test the full 25-phrase UI flow before real components are ready:

```python
SCRIPT = [
    "Hi", "welcome", "to", "our", "project", "a",
    "sign language interpreter", "that", "interprets",
    "American Sign Language", "into", "text", "and", "speech",
    "it", "uses", "computer vision", "and", "machine learning",
    "to", "detect", "hand", "gestures", "and",
    "converts them into text and speech"
]

mock_index = [0]

def mock_pipeline(frame):
    mock_pipeline.calls = getattr(mock_pipeline, 'calls', 0) + 1
    if mock_pipeline.calls % 30 == 0 and mock_index[0] < len(SCRIPT):
        phrase = SCRIPT[mock_index[0]]
        mock_index[0] += 1
        return {
            'frame': frame,
            'prediction': {
                'sign': phrase, 'confidence': 1.0,
                'script_index': mock_index[0], 'total': len(SCRIPT),
                'advanced': True, 'completed': mock_index[0] >= len(SCRIPT),
            }
        }
    return {'frame': frame, 'prediction': None}
```

Run `test_gradio_app()` and visually confirm all 25 phrases display in order.

---

## Task 3 — Demo Screen Layout

Target layout for the recording:

```
┌──────────────────────────────────────────────────────┐
│  🤟 ASL Sign Language Interpreter                    │
├──────────────────────────┬───────────────────────────┤
│  📹 Live Camera Feed     │  Detected Sign            │
│  [hand tracking overlay] │  ┌─────────────────────┐  │
│                          │  │  sign language       │  │
│                          │  │  interpreter        │  │
│                          │  └─────────────────────┘  │
│                          │  Phrase 7 of 25            │
│                          │  Confidence: 100%          │
│                          │                            │
│                          │  [↺ Reset Script]          │
└──────────────────────────┴───────────────────────────┘
```

---

## Integration Interface

### Input (from Person 4 / pipeline)

```python
pipeline_func  # callable: frame → {frame, prediction}
tts_engine     # TTSEngine instance with .speak(text)
```

### Output (to screen recording)

- Live annotated video feed
- Large current-phrase display
- Phrase progress counter
- Audio on each sign detection

---

## Success Criteria

- Gradio interface loads and shows webcam feed
- Current phrase displays prominently on sign detection
- Progress indicator shows correct phrase number (e.g. "Phrase 7 of 25")
- TTS audio plays for each phrase
- Reset button restarts script cleanly
- UI looks clean and professional in a screen recording

---

## Tips

1. Test with the mock pipeline first — don't wait for other components.
2. The **current phrase display is the #1 priority** — make it large and clear.
3. The Reset button is critical — the signer must be able to restart without
   touching the code.
4. Keep the layout simple: two columns, video left, info right.
5. Coordinate with Person 4 to confirm the pipeline function signature.
