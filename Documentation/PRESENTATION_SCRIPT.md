# ASL Real-Time Interpreter -- Presentation Script

> **Format:** 3-person team presentation with live demo.
> **Duration:** ~8-10 minutes total.
> **Setup:** Laptop with webcam, browser open to `localhost:7860`, projector/screen share active.

---

## Section 1: Opening Hook

**Speaker: Team Lead**

> "Imagine you're deaf or hard of hearing, and you walk into a room where nobody understands sign language. You sign 'Hello' -- and nobody responds. That's the daily reality for over 70 million people worldwide who use sign language as their primary form of communication.
>
> Today we're going to show you something we built in under two hours -- a real-time sign language interpreter that watches your hands through a webcam, understands what you're signing, displays it as text, and speaks it out loud. No paid APIs, no cloud dependency, no subscription. Completely free, running entirely on your local machine."

---

## Section 2: The Problem

**Speaker: Team Lead**

> "The communication barrier between sign language users and non-signers is one of the most overlooked accessibility challenges in tech. Professional ASL interpreters cost between $50 and $150 per hour. They need to be booked in advance. They're not available at 2 AM when you need to call an ambulance.
>
> Existing solutions either require expensive cloud APIs with per-request pricing, need powerful GPUs, or simply don't work in real-time. We asked ourselves: can we build something that's fast, free, and works right now -- on any laptop with a webcam?"

---

## Section 3: Architecture Overview

**Speaker: CV / ML Developer**

> "Let me walk you through how the system works. The architecture is a linear pipeline with four stages."

**[Show architecture diagram or slide]**

```
Webcam Frame
    |
    v
MediaPipe GestureRecognizer
    |  (21 hand landmarks + gesture label + confidence)
    v
ASL Sign Mapper (rule-based classification)
    |  (maps landmarks to ASL meanings)
    v
Gesture Stabilizer (temporal voting buffer)
    |  (locks in a gesture only when 5 of 8 frames agree)
    v
Output: Gradio UI (text + skeleton overlay) + macOS TTS (voice)
```

> "**Stage 1 -- Hand Detection.** Every frame from the webcam is fed into Google's MediaPipe GestureRecognizer. This is an on-device, pre-trained model that extracts 21 three-dimensional landmarks from the hand -- fingertips, knuckles, wrist -- along with a gesture classification and confidence score. It runs on CPU with Metal GPU acceleration, no CUDA required.
>
> **Stage 2 -- ASL Sign Mapping.** MediaPipe only knows 7 generic gestures like 'Open Palm' or 'Closed Fist.' That's not enough for real ASL. So we built a rule-based classifier on top of the 21 landmarks. It computes which fingers are extended, the thumb direction, and finger combinations to recognise 13 distinct ASL signs -- things like 'I Love You,' 'Peace,' 'Yes,' 'No,' numbers one through four, and more. If our rules don't match, we fall back to MediaPipe's gesture as a safety net.
>
> **Stage 3 -- Temporal Stabilization.** Raw frame-by-frame predictions are noisy. Your hand might read as 'Peace' for 6 frames, then 'Three' for 1 frame due to slight finger movement. We built a voting buffer that only changes the displayed gesture when at least 5 out of the last 8 frames agree on the same label. This eliminates jitter and gives a stable, confident output.
>
> **Stage 4 -- Output.** The recognised sign is displayed on screen alongside the live video feed with a hand skeleton overlay drawn using OpenCV. Simultaneously, macOS's native speech synthesiser speaks the sign aloud. If the gesture changes mid-speech, the system interrupts the current utterance and immediately starts speaking the new one."

---

## Section 4: Design Decisions -- Why No LLM APIs

**Speaker: Integration Specialist**

> "One of the most important decisions we made was what we chose *not* to use. Let me explain the trade-offs we considered."

### Decision 1: No Cloud LLM APIs (OpenAI, Anthropic, Google)

> "We deliberately chose not to use any paid LLM APIs in the final application. Here's why:
>
> **Cost barrier.** If this tool is meant for people with disabilities, it has to be free. Period. API calls to GPT-4, Claude, or Gemini cost money per request. A real-time system processing 25 frames per second would burn through API credits in minutes. We refuse to build an accessibility tool that requires a credit card.
>
> **Latency.** Cloud API round-trips add 200-500ms per call. For real-time sign language interpretation, that delay is unacceptable. Our entire pipeline runs in under 200ms end-to-end because everything runs locally.
>
> **Privacy.** Sign language users are communicating personal, potentially sensitive information. Sending every webcam frame to a cloud server raises serious privacy concerns. Our system never transmits a single frame outside your machine.
>
> **Offline capability.** Internet goes down. Cloud services have outages. A deaf person's ability to communicate should never depend on an API's uptime."

### Decision 2: Rule-Based Classification Over Training a Deep Learning Model

> "We also chose a rule-based approach for ASL sign classification rather than training a custom neural network. With a 2-hour hackathon constraint, collecting and labelling thousands of training images per sign wasn't feasible. Instead, we leverage the 21 high-quality landmarks that MediaPipe already provides and write geometric rules -- 'if the thumb, index, and pinky are extended but the middle and ring fingers are closed, that's I Love You.' This is interpretable, debuggable, and runs in microseconds.
>
> The model is trained on a vocabulary of 13 ASL signs and numbers, with the MediaPipe gesture recognizer providing an additional 7-gesture safety net."

### Decision 3: On-Device TTS Instead of Cloud Speech Services

> "For text-to-speech, we use the operating system's built-in speech synthesiser rather than services like ElevenLabs or Google Cloud TTS. It's not the most natural-sounding voice -- but it's instant, free, works offline, and never fails. For an accessibility tool, reliability beats polish."

### What We Did Use Claude For

> "To be transparent -- we did use Claude extensively as a development tool during the hackathon. Claude helped us architect the system, debug MediaPipe integration issues, and write code faster. But Claude is not in the runtime path. Once the application is built, it runs with zero external dependencies. Claude was our pair programmer, not our production dependency."

---

## Section 5: Technical Challenges

**Speaker: UI Developer**

> "Building this in two hours meant hitting problems fast. Here are the three biggest ones we solved."

### Challenge 1: MediaPipe API Breaking Changes

> "MediaPipe recently removed its legacy `mp.solutions` Python API. Our initial hand detector code was built against the old API and broke immediately. We had to pivot to the new `mediapipe.tasks` API on the fly, restructuring how landmarks flow through the pipeline. The gesture recognizer now handles both detection and landmark extraction in a single pass."

### Challenge 2: UI Stability at 25 FPS

> "Streaming webcam frames through a Gradio web interface at 25 frames per second caused severe flickering -- the detection panel would flash different gestures every millisecond. We solved this with three techniques: a temporal voting buffer (the stabilizer we mentioned), diff-gated HTML updates that only push to the browser when content actually changes, and CSS optimisations that suppress Gradio's built-in loading animations."

### Challenge 3: Text-to-Speech Hanging on macOS

> "Python's `pyttsx3` library -- the standard cross-platform TTS engine -- has a known bug on macOS where it hangs after the first utterance. It would speak 'Hello' perfectly, then go completely silent for every subsequent gesture. We bypassed it entirely and shell out to macOS's native `say` command via subprocess. When a new gesture is detected, we kill any in-progress speech and start the new one immediately. Zero gestures get skipped."

---

## Section 6: Live Demo

**Speaker: Team Lead**

> "Enough talking -- let me show you."

**[Switch to browser showing localhost:7860]**

**Demo sequence:**

1. **Show empty state** -- "Here's the application. Clean dark interface, waiting for camera input."
2. **Enable webcam** -- "I'll start the camera. You can see the live feed on the left."
3. **Hold up open hand** -- "Open palm -- the system detects 'Hello.' You can see the hand skeleton tracking my landmarks in real time, the bounding box, and it just spoke 'Hello' out loud."
4. **Make a fist** -- "Closed fist -- 'Yes.' Notice how it interrupted the previous word and immediately said the new one."
5. **Thumbs up** -- "'Good.' The confidence score is shown in the detection panel."
6. **Peace sign** -- "'Peace.' Watch how stable the detection is -- no flickering."
7. **I Love You sign** (thumb + index + pinky) -- "'I Love You.' That's our rule-based classifier recognising a specific ASL sign that MediaPipe alone wouldn't understand."
8. **Thumbs down** -- "'Bad.'"
9. **Remove hand from frame** -- "And when I take my hand away, it correctly shows 'Waiting for hand...' -- no false positives."

> "That's 13 distinct ASL signs, detected in real-time, spoken aloud, with zero cloud dependencies."

---

## Section 7: Supported Signs Reference

> "Here's the full vocabulary our model recognises."

| Hand Pose | ASL Meaning | Detection Method |
|---|---|---|
| All fingers open | Hello | Rule-based |
| Closed fist | Yes | Rule-based |
| Thumb up | Good | Rule-based |
| Thumb down | Bad | Rule-based |
| Thumb + index + pinky | I Love You | Rule-based |
| Index + middle (no thumb) | No | Rule-based |
| Index + middle (+ thumb) | Peace | Rule-based |
| Index only | One | Rule-based |
| Index + middle + ring | Three | Rule-based |
| Four fingers (no thumb) | Four | Rule-based |
| Thumb + pinky | Phone | Rule-based |
| Pinky only | I | Rule-based |
| Index + pinky | Rock On | Rule-based |

---

## Section 8: Future Vision

**Speaker: Team Lead**

> "What we built today is a proof of concept -- but the path to a production tool is clear.
>
> **Expanded vocabulary.** Our rule-based system handles static poses. The next step is training a sequence model -- an LSTM or Transformer -- on video clips to recognise dynamic signs that involve motion, like 'Thank You' or 'Sorry.'
>
> **Two-handed signs.** ASL uses both hands extensively. MediaPipe supports multi-hand tracking; we just need to extend our classifier.
>
> **Facial grammar.** In ASL, facial expressions change meaning. A raised eyebrow turns a statement into a question. Integrating MediaPipe's face mesh would capture this.
>
> **Mobile deployment.** MediaPipe runs on Android and iOS natively. A mobile app would make this accessible anywhere.
>
> **Bidirectional communication.** Right now we translate ASL to speech. The reverse -- speech to sign language animation -- would complete the communication loop.
>
> But even as a two-hour hackathon prototype, this demonstrates something important: the technology to break communication barriers exists today, it can run on commodity hardware, and it doesn't have to cost a thing."

---

## Section 9: Closing

**Speaker: Team Lead**

> "We built this because we believe accessibility tools shouldn't be locked behind paywalls. Every design decision -- local processing, rule-based classification, native TTS -- was made with one principle: if someone needs this to communicate, it should just work. No API key, no subscription, no internet required.
>
> Thank you."

---

## Quick Reference: Key Numbers for Q&A

| Metric | Value |
|---|---|
| Build time | ~2 hours |
| Team size | 4 people |
| End-to-end latency | < 200ms |
| Frame rate | 25 FPS |
| Recognised signs | 13 distinct ASL signs |
| External API calls at runtime | 0 |
| Cost to run | Free |
| Internet required | No |
| Languages | Python (backend), HTML/CSS/JS (Gradio UI) |
| Model | MediaPipe GestureRecognizer (pre-trained, on-device) |
| TTS engine | macOS native `say` command |
| Stabilization | 8-frame voting buffer, 5/8 majority threshold |
