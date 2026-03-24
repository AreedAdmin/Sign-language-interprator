# ASL Real-Time Interpreter

## What It Does

A real-time American Sign Language interpreter that runs entirely on-device. Point your webcam at a hand gesture, and the system detects the sign, displays it as text with a live hand skeleton overlay, and speaks it aloud -- all in under 200ms, with zero cloud dependencies.

## How We Built It

We wired a four-stage local pipeline in Python:

1. **MediaPipe GestureRecognizer** extracts 21 hand landmarks per frame at 25 FPS
2. **Rule-based ASL classifier** maps finger geometry (extension ratios, thumb direction, finger combinations) to 13 ASL signs -- extending MediaPipe's 7 built-in gestures with custom landmark rules
3. **Temporal stabilizer** uses a majority-vote buffer (\(k \geq 5\) out of \(n = 8\) frames) to suppress per-frame jitter
4. **Gradio UI + native macOS TTS** renders the annotated video feed and speaks each new gesture via `subprocess.Popen`, interrupting in-flight speech on gesture change

No LLM APIs are called at runtime. Claude was used purely as a development tool for architecture and debugging.

## Challenges

- **MediaPipe breaking changes** -- the `mp.solutions` API was removed in our installed version; we pivoted to `mediapipe.tasks` mid-build and extracted landmarks directly from the GestureRecognizer
- **UI flicker at 25 FPS** -- solved with diff-gated HTML updates, CSS animation suppression, and the temporal voting buffer
- **TTS hanging on macOS** -- `pyttsx3` deadlocks after the first utterance on Apple Silicon; we bypassed it entirely with the native `say` command and a kill-and-replace subprocess strategy

## What We Learned

Building for accessibility forces better engineering. The constraint of "no paid APIs" pushed us toward a faster, more private, more reliable system than we would have built otherwise. Rule-based classification on high-quality landmarks is surprisingly effective -- 13 signs with zero training data, just geometry.
