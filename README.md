<div align="center">

# 🤟 ASL Real-Time Interpreter

### American Sign Language → Text → Speech, in real time.

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Claude](https://img.shields.io/badge/Claude-191919?style=for-the-badge&logo=anthropic&logoColor=white)](https://claude.ai/)
[![pyttsx3](https://img.shields.io/badge/pyttsx3%20TTS-4CAF50?style=for-the-badge&logo=audiomack&logoColor=white)](https://pyttsx3.readthedocs.io/)

<br/>

> Built in 2 hours as a hackathon project — a fully functional real-time sign language interpreter  
> that captures webcam video, detects hand landmarks, classifies ASL signs, and speaks them aloud.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [ASL Vocabulary](#asl-vocabulary)
- [Configuration](#configuration)
- [Team](#team)
- [Future Enhancements](#future-enhancements)

---

## Overview

The **ASL Real-Time Interpreter** is a web application that bridges the communication gap for ASL (American Sign Language) users. A signer performs hand signs in front of their webcam; the system detects each sign using Google's MediaPipe Gesture Recognizer, outputs the corresponding phrase as text on screen, and speaks it aloud using a non-blocking text-to-speech engine — all in real time inside a Gradio web interface.

**Demo format:** A scripted 25-sign sequence. The signer performs each sign in order; the system advances through a pre-defined phrase list (`asl_words.txt`) one phrase at a time, displaying and speaking each phrase as it's detected. This design eliminates the need for training a custom ML model from scratch — perfect for a hackathon time constraint.

**Key characteristics:**

| Property | Value |
|---|---|
| End-to-end latency target | < 200ms |
| Target frame rate | 30 FPS |
| Vocabulary size | 25 words / phrases |
| Classifier mode | Script / Sequence Mode |
| Demo format | Screen-recorded webcam session |

---

## Demo

```
🖥️  Browser — localhost:7860   (screen-recorded)
┌─────────────────────────────────────────────────────────────┐
│  📹 Live Webcam Feed          📊 Script Output Panel        │
│  [Signer performing ASL]      Current: "sign language       │
│  [Hand tracking overlay]       interpreter"                 │
│                               Phrase: 7 of 25               │
│                               Confidence: 91%               │
│  🔊 Audio plays automatically  📝 Full transcript shown     │
│     on each sign detection         with current phrase      │
└─────────────────────────────────────────────────────────────┘
```

**Recording flow:**

1. Signer opens `localhost:7860` in a browser and starts a screen recording.
2. The webcam activates and shows the live video feed.
3. Signer performs sign #1 → system outputs **"Hi"**, TTS speaks it aloud.
4. Signer performs sign #2 → system outputs **"welcome"**, TTS speaks it.
5. Sequence continues through all 25 phrases.
6. Final phrase **"converts them into text and speech"** is spoken.
7. Screen recording is saved as the demo video.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core runtime and application logic |
| **Gradio ≥ 4.0** | Instant web UI — video input, text display, audio controls |
| **MediaPipe ≥ 0.10** | Google's pre-built Gesture Recognizer — 21 hand landmarks per frame |
| **OpenCV ≥ 4.8** | Frame capture and BGR↔RGB conversion for the vision pipeline |
| **pyttsx3 ≥ 2.90** | Offline, non-blocking text-to-speech synthesis (no API key needed) |
| **NumPy ≥ 1.24** | Numerical processing for landmark arrays |
| **Claude (Anthropic)** | AI-assisted development — architecture planning, code generation, and debugging throughout the hackathon |
| **python-dotenv** | Environment variable management |

---

## Architecture

The processing pipeline is strictly linear:

```
Webcam
  │
  ▼
HandDetector (MediaPipe GestureRecognizer)
  │  ← 21 (x, y, z) landmarks + confidence score
  ▼
ASLClassifier (Script/Sequence Mode)
  │  ← fires current phrase when a confident sign is held
  ▼
ASLInterpreterApp  ──→  TTSEngine (pyttsx3, async background thread)
  │
  ▼
Gradio UI (localhost:7860)
  ├── Live video feed with hand-tracking overlay
  ├── Current phrase display + confidence bar
  ├── Progress indicator (e.g. "Phrase 7 of 25")
  └── Full script transcript panel
```

### Component Responsibilities

| Module | File | Responsibility |
|---|---|---|
| **Hand Detection** | `src/models/hand_detector.py` | MediaPipe Hands — 21 landmarks per hand, confidence scoring, temporal smoothing |
| **ASL Classification** | `src/models/asl_classifier.py` | Script-mode classifier — advances through `asl_words.txt` each time a confident sign is detected |
| **Text-to-Speech** | `src/models/tts_engine.py` | Non-blocking pyttsx3 audio synthesis via a background worker thread |
| **UI** | `src/ui/gradio_app.py` | Gradio interface — video stream, sign history, confidence display, audio controls |
| **Entry Point** | `src/main.py` | `ASLInterpreterApp` wires all components together |
| **Integration Utilities** | `src/utils/integration.py` | Mock components (`MockHandDetector`, `MockASLClassifier`, `MockTTSEngine`) for isolated testing |

### Interface Contracts

**Hand Detector → ASL Classifier:**
```python
{
    'landmarks':    [(x1,y1,z1), ..., (x21,y21,z21)],  # 21 MediaPipe points
    'hand_present': bool,
    'confidence':   float,
    'timestamp':    float
}
```

**ASL Classifier → UI / TTS:**
```python
{
    'sign':       'sign language interpreter',  # Current phrase from asl_words.txt
    'confidence': 0.91,                         # MediaPipe gesture score
    'top_3':      [('Thumb_Up', 0.91), ...],    # Top gesture candidates
    'index':      6                             # Position in the 25-phrase script
}
```

---

## Project Structure

```
sign-language-interprator/
├── src/
│   ├── main.py                    # Application entry point & pipeline orchestration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hand_detector.py       # MediaPipe hand landmark detection
│   │   ├── asl_classifier.py      # Script-mode ASL classifier
│   │   └── tts_engine.py          # Non-blocking TTS engine
│   ├── ui/
│   │   ├── __init__.py
│   │   └── gradio_app.py          # Gradio web interface
│   └── utils/
│       ├── __init__.py
│       └── integration.py         # Mock components & shared utilities
├── data/
│   └── models/
│       └── gesture_recognizer.task  # MediaPipe pre-trained model file
├── tests/
│   └── test_integration.py        # Integration test suite
├── instructions/                  # Per-role implementation guides
│   ├── README.md
│   ├── person1_ml_expert.md
│   ├── person2_cv_developer.md
│   ├── person3_ui_developer.md
│   └── person4_integration_specialist.md
├── asl_words.txt                  # The 25-phrase demo script
├── requirements.txt
├── .env
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A working webcam
- macOS / Linux / Windows with audio output

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/sign-language-interprator.git
cd sign-language-interprator

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python src/main.py
```

The Gradio interface launches at **[http://localhost:7860](http://localhost:7860)**.

### Running Tests

```bash
python tests/test_integration.py
```

---

## How It Works

### 1. Gesture Detection — MediaPipe GestureRecognizer

Each video frame is passed through Google's pre-trained `gesture_recognizer.task` model (stored in `data/models/`). The recognizer returns:

- A named gesture label (e.g. `Thumb_Up`, `Open_Palm`, `Victory`)
- A confidence score between 0.0 and 1.0

Frames where no gesture is detected, or where the top gesture is labelled `None`, are discarded.

### 2. Script-Mode Classification

Rather than training a custom model to recognise specific ASL signs, the classifier operates in **Script Mode**:

1. Any frame where MediaPipe returns a gesture with confidence ≥ **0.5** counts as "a sign was performed."
2. A **1.5-second cooldown** prevents the same sign from advancing the script multiple times.
3. On each successful detection, the classifier emits the *next phrase* from `asl_words.txt` and increments its internal index.
4. Calling `classifier.reset()` restarts from phrase #1 — useful when a recording take goes wrong.

This design means the signer rehearses 25 signs mapped to the script, and the system advances in sync with their performance.

### 3. Non-Blocking Text-to-Speech

`TTSEngine` runs `pyttsx3` inside a **background daemon thread**, draining a `Queue`. The main video pipeline never blocks waiting for audio — `speak()` returns instantly, and the audio plays asynchronously.

```python
tts.speak("sign language interpreter")          # queues, returns immediately
tts.speak("American Sign Language", priority=True)  # clears queue, speaks next
```

### 4. Gradio Interface

The Gradio app (`src/ui/gradio_app.py`) provides:

- **Live video feed** — annotated with hand tracking overlay
- **Current phrase display** — large, prominently centred text
- **Progress indicator** — e.g. `Phrase 7 of 25`
- **Confidence display** — percentage score from MediaPipe
- **Full transcript panel** — all 25 phrases with the current one highlighted
- **Reset button** — instantly restarts the script for a new take

---

## ASL Vocabulary

The 25-phrase demo script (`asl_words.txt`) forms a complete sentence when read aloud:

> *Hi, welcome to our project — a sign language interpreter that interprets American Sign Language into text and speech. It uses computer vision and machine learning to detect hand gestures and converts them into text and speech.*

| # | Phrase |
|---|---|
| 1 | Hi |
| 2 | welcome |
| 3 | to |
| 4 | our |
| 5 | project |
| 6 | a |
| 7 | sign language interpreter |
| 8 | that |
| 9 | interprets |
| 10 | American Sign Language |
| 11 | into |
| 12 | text |
| 13 | and |
| 14 | speech |
| 15 | it |
| 16 | uses |
| 17 | computer vision |
| 18 | and |
| 19 | machine learning |
| 20 | to |
| 21 | detect |
| 22 | hand |
| 23 | gestures |
| 24 | and |
| 25 | converts them into text and speech |

---

## Configuration

Key tuning parameters in `src/models/asl_classifier.py`:

| Constant | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum gesture confidence to count as a sign |
| `COOLDOWN_SECONDS` | `1.5` | Seconds to wait before the next phrase can fire |

TTSEngine settings in `src/models/tts_engine.py`:

| Parameter | Default | Description |
|---|---|---|
| `rate` | `150` | Speech rate (words per minute) |
| `volume` | `0.9` | Volume (0.0 – 1.0) |

---

## Team

This project was built by a 4-person team in 2 hours:

| Role | Responsibility |
|---|---|
| **ML Expert (Lead)** | ASL classifier, TTS integration, pipeline coordination |
| **Computer Vision Developer** | MediaPipe hand detection and landmark extraction |
| **UI / Frontend Developer** | Gradio interface, visual design, real-time text updates |
| **Integration & Testing Specialist** | Project setup, component integration, end-to-end testing |

---

## Future Enhancements

- **Extended vocabulary** — expand beyond the 25-phrase script to free-form ASL recognition
- **Multi-hand support** — two-handed sign recognition for more complex signs
- **Facial expression integration** — incorporate facial grammar markers for full ASL semantics
- **Mobile app** — React Native or Flutter client consuming a FastAPI backend
- **Cloud deployment** — containerised deployment on AWS / GCP for public access
- **Higher-quality TTS** — swap `pyttsx3` for ElevenLabs or Google Cloud TTS for more natural voice output
- **Custom gesture training** — fine-tune the MediaPipe model on a domain-specific ASL dataset

---

<div align="center">

Built with ❤️ at a hackathon · Powered by [MediaPipe](https://mediapipe.dev/), [Gradio](https://www.gradio.app/), and [Claude](https://claude.ai/)

</div>
