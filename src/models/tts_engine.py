import threading
import time
from queue import Queue, Empty

import pyttsx3


class TTSEngine:
    """
    Non-blocking text-to-speech engine.

    Runs a background worker thread so that speak() returns immediately
    and audio plays asynchronously — won't block the video pipeline.

    Usage:
        tts = TTSEngine()
        tts.speak("HELLO")          # queues, returns at once
        tts.speak("YES", priority=True)  # clears queue, says this next
        tts.stop()                  # clean shutdown
    """

    def __init__(self, rate: int = 150, volume: float = 0.9):
        self._queue: Queue[str] = Queue()
        self._stop_event = threading.Event()
        self._speaking = False

        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', rate)
        self._engine.setProperty('volume', volume)

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[TTSEngine] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, priority: bool = False) -> None:
        """
        Queue text for speech.

        Args:
            text:     The string to speak.
            priority: If True, clears any pending items first so this
                      plays as soon as the current utterance finishes.
        """
        if not text:
            return
        if priority:
            self._drain_queue()
        self._queue.put(text)

    def is_busy(self) -> bool:
        """True while speech is in progress or items are queued."""
        return self._speaking or not self._queue.empty()

    def stop(self) -> None:
        """Stop speaking and shut down the worker thread."""
        self._drain_queue()
        self._stop_event.set()
        try:
            self._engine.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._speaking = True
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[TTSEngine] Speech error: {e}")
            finally:
                self._speaking = False
                self._queue.task_done()


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------

def test_tts():
    tts = TTSEngine()
    print("Speaking: 'Hello, ASL interpreter is ready'")
    tts.speak("Hello, ASL interpreter is ready")

    # Wait for audio to finish
    while tts.is_busy():
        time.sleep(0.1)

    tts.speak("YES")
    while tts.is_busy():
        time.sleep(0.1)

    print("TTS test complete.")
    tts.stop()


if __name__ == '__main__':
    test_tts()
