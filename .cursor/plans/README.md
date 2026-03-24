# ASL Real-time Interpreter - Project Plans

This directory contains all planning documents for the ASL (American Sign Language) real-time interpreter project.

## Plan Files

### Main Project Plan
- **[asl_real-time_interpreter_63183d10.plan.md](asl_real-time_interpreter_63183d10.plan.md)** - Complete project plan with 2-hour hackathon timeline and team delegation strategy

### Individual Team Member Plans
- **[person1_ml_expert.md](person1_ml_expert.md)** - Detailed tasks for ML Expert (Team Lead)
- **[person2_cv_developer.md](person2_cv_developer.md)** - Detailed tasks for Computer Vision Developer
- **[person3_ui_developer.md](person3_ui_developer.md)** - Detailed tasks for UI/Frontend Developer
- **[person4_integration_specialist.md](person4_integration_specialist.md)** - Detailed tasks for Integration & Testing Specialist

## Quick Reference

### Project Overview
Build a real-time ASL interpreter that:
1. Captures video from webcam
2. Detects hand landmarks using MediaPipe
3. Classifies ASL signs using ML model
4. Displays detected text in real-time
5. Converts text to speech using TTS

### Timeline: 2 Hours
- **Hour 1 (0-60 min)**: Parallel development of components
- **Hour 2 (60-120 min)**: Integration, testing, and demo preparation

### Team Structure (4 People)
1. **Person 1**: ML Expert (ASL classifier + TTS)
2. **Person 2**: Computer Vision (MediaPipe hand detection)
3. **Person 3**: UI Developer (Gradio interface)
4. **Person 4**: Integration Specialist (project setup + testing)

### Success Criteria
- ✅ Live video feed with hand detection overlay
- ✅ Recognition of 7-10 basic ASL signs
- ✅ Real-time text display of detected signs
- ✅ Text-to-speech audio output
- ✅ Working Gradio web interface
- ✅ End-to-end processing under 1 second

## Getting Started

1. Read the main project plan: `asl_real-time_interpreter_63183d10.plan.md`
2. Each team member should read their individual plan
3. Set up the development environment following Person 4's setup guide
4. Start parallel development according to the timeline

## Architecture

```
Camera Input → Hand Detection → ASL Classification → Text Display → Text-to-Speech
     ↓              ↓                ↓                 ↓            ↓
  Webcam      MediaPipe        ML Model         Gradio UI      Audio Output
```

## Technology Stack
- **Computer Vision**: MediaPipe Hands
- **Machine Learning**: Pre-trained ASL model or rule-based classifier
- **Frontend**: Gradio web interface
- **Text-to-Speech**: pyttsx3
- **Language**: Python 3.8+