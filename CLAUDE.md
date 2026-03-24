# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time ASL (American Sign Language) interpreter that captures webcam video, detects hand landmarks via MediaPipe, classifies signs via ML, displays text, and synthesizes speech via TTS. Built as a 2-hour hackathon project.

**Demo format:** Live webcam, screen-recorded for the presentation — the signer performs a scripted 25-sign sequence and the system outputs each phrase from `asl_words.txt` in order.

**Classifier mode:** Script/Sequence Mode — the classifier advances through `asl_words.txt` sequentially each time a confident sign is detected. No free-form recognition needed.

**Target vocabulary (exact lines from `asl_words.txt`):**
```
Hi, welcome, to, our, project, a, sign language interpreter, that,
interprets, American Sign Language, into, text, and, speech, it, uses,
computer vision, and, machine learning, to, detect, hand, gestures,
and, converts them into text and speech
```

## Setup & Running

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py               # Launches Gradio UI at http://localhost:7860
```

## Testing

```bash
python tests/test_integration.py
```

## Architecture

The pipeline is linear:

```
Webcam → HandDetector → ASLClassifier → ASLInterpreterApp → Gradio UI + TTSEngine
```

| Module | File | Responsibility |
|--------|------|----------------|
| Hand Detection | `src/models/hand_detector.py` | MediaPipe Hands — 21 landmarks per hand, confidence scoring, temporal smoothing |
| ASL Classification | `src/models/asl_classifier.py` | Sign recognition from landmark sequences; rule-based or TF/PyTorch model |
| Text-to-Speech | `src/models/tts_engine.py` | pyttsx3 audio synthesis |
| UI | `src/ui/gradio_app.py` | Gradio web interface — video stream, sign history, confidence display, audio controls |
| Entry point | `src/main.py` | `ASLInterpreterApp` wires all components together |
| Integration helpers | `src/utils/integration.py` | Shared utilities and mock components for testing |

**Performance targets:** Real-time webcam feed — target <200ms end-to-end latency, 30 FPS.

## Key Design Decisions

- **Mock components** (`MockHandDetector`, `MockASLClassifier`, `MockTTSEngine`) in `src/utils/integration.py` allow each component to be developed and tested independently.
- **Interface contracts** are strict: `HandDetector` outputs a list of 21 (x, y, z) landmark tuples; `ASLClassifier` consumes that list and returns a sign label + confidence float.
- **Script vocabulary** is in `asl_words.txt` (25 words/phrases). The classifier steps through this list in order — one phrase output per detected sign.
- **Low demo risk** — if a take goes wrong, just reset the classifier (`classifier.reset()`) and sign again.
- The classifier can be swapped between rule-based (no model file required) and a trained TF/PyTorch model by changing the backend in `asl_classifier.py`.

## Planning Docs

Detailed per-person implementation plans (code skeletons, edge cases, backup strategies) live in `.cursor/plans/`:
- `asl_real-time_interpreter_63183d10.plan.md` — full architecture and Gantt chart
- `person1_ml_expert.md` — ML model + TTS
- `person2_cv_developer.md` — MediaPipe hand detection
- `person3_ui_developer.md` — Gradio UI
- `person4_integration_specialist.md` — project setup, integration, testing, `requirements.txt`
