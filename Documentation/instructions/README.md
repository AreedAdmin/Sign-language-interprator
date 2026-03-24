# Agent Instructions

This folder contains the task instructions for each AI agent (team member) working on the ASL Real-time Interpreter project.

## Demo Format

The demo is a **pre-recorded video** — not a live presentation.
The signer performs a scripted 25-sign sequence. The system outputs each phrase from `asl_words.txt` in order. Re-record until the take is clean.

## Scripted Speech (from `asl_words.txt`)

> *"Hi, welcome to our project — a sign language interpreter that interprets
> American Sign Language into text and speech. It uses computer vision and
> machine learning to detect hand gestures and converts them into text and speech."*

## Key Design Decision: Script Mode

The classifier does **not** identify which specific ASL sign is made.
It simply detects that a hand sign was held for a minimum number of frames,
then advances to the next phrase in `asl_words.txt`. This eliminates all ML
training complexity and guarantees reliable output for the recording.

## Team Files

| File | Role |
|------|------|
| [person1_ml_expert.md](person1_ml_expert.md) | ASL classifier (script mode) + TTS engine |
| [person2_cv_developer.md](person2_cv_developer.md) | MediaPipe hand detection |
| [person3_ui_developer.md](person3_ui_developer.md) | Gradio web interface |
| [person4_integration_specialist.md](person4_integration_specialist.md) | Project setup, integration, testing |

## Architecture

```
Webcam → HandDetector → ASLClassifier (script mode) → TTSEngine
                                   ↓
                            ASLGradioApp (UI)
```

## Full Planning Docs

Detailed architecture, Gantt charts, and per-component deep dives live in
`.cursor/plans/`.
