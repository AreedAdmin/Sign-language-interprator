# ASL Real-Time Interpreter

## What It Does

Sign "I Love You" into your laptop camera. In under 200 milliseconds -- faster than a blink -- the words appear on screen, a skeleton traces your hand, and a voice says it out loud. No internet. No API key. No cost. Just your webcam and a Python script.

We built a real-time American Sign Language interpreter that recognises 13 ASL signs, translates them to text, and speaks them aloud -- running entirely on your local machine with zero cloud dependencies. Because if someone needs this tool to communicate, it shouldn't require a credit card.

## How We Built It

Four people. Two hours. One pipeline:

```
Webcam --> MediaPipe (21 hand landmarks) --> ASL Sign Mapper --> Stabilizer --> Voice + UI
```

**The hands.** Google's MediaPipe GestureRecognizer extracts 21 three-dimensional landmarks from each hand -- every knuckle, every fingertip, the wrist -- at 25 frames per second, running on-device with Metal GPU acceleration.

**The brain.** MediaPipe only knows 7 generic gestures. ASL has thousands. So we wrote a geometry engine on top of the raw landmarks: compute which fingers are extended, measure thumb direction, match finger combinations against known ASL patterns. The math is simple -- if \(\frac{\| \text{tip} - \text{wrist} \|}{\| \text{mcp} - \text{wrist} \|} > 1.1\), the finger is extended. Compose five of those booleans and you can distinguish "I Love You" (thumb + index + pinky) from "Rock On" (index + pinky) from "Phone" (thumb + pinky). 13 signs, zero training data, just geometry.

**The filter.** Raw predictions flicker. A single noisy frame can flip "Peace" to "Three" and back. We built a temporal voting buffer: a gesture only locks in when \(\geq 5\) out of the last 8 frames agree. Jitter gone.

**The voice.** When a new stable gesture appears, macOS's native `say` command speaks it in a background thread. If the gesture changes mid-word, we kill the running process and start the new one immediately. Every sign gets spoken. Nothing gets skipped.

**The interface.** A Gradio web app with a live camera feed, hand skeleton overlay drawn in OpenCV, real-time detection panel, sign history, and sentence builder -- all dark-themed and optimised to feel responsive at 25 FPS.

No LLM APIs are called at runtime. We used Claude as our AI pair-programmer during the hackathon, but it lives nowhere in the deployed system.

## Challenges

**The API that vanished.** Halfway through the build, we discovered MediaPipe had silently removed its `mp.solutions` Python API in the version we installed. Our entire hand detection module broke. We rearchitected on the spot -- pulling landmarks directly from the GestureRecognizer's internal results and drawing the skeleton ourselves with raw OpenCV calls.

**The flickering UI.** Streaming webcam frames through a browser at 25 FPS made Gradio's interface strobe like a nightclub. Detection results flashed different gestures every millisecond. We tamed it with three layers: the voting buffer for gesture stability, diff-gated HTML updates that only push to the browser when content actually changes, and surgical CSS to suppress Gradio's built-in loading animations.

**The voice that spoke once and died.** Python's `pyttsx3` TTS library has a silent, undocumented deadlock on Apple Silicon -- it speaks the first word perfectly, then freezes forever. We ripped it out and replaced it with direct subprocess calls to macOS `say`, using a kill-and-replace strategy so new gestures interrupt old speech instantly.

## What We Learned

The best constraint we imposed on ourselves was "no paid APIs." It wasn't a limitation -- it was a forcing function. It pushed us toward a system that's faster (no network round-trips), more private (no frames leave your machine), more reliable (no API outages), and more accessible (no cost barrier) than anything we would have built with a cloud-first approach.

And the biggest surprise: you don't always need a neural network. Twenty-one high-quality landmarks and some finger-distance ratios gave us 13 reliable ASL signs with literally zero training data. Sometimes the right abstraction is just math.
