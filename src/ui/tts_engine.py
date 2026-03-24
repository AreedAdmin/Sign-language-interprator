"""
Thread-safe Text-to-Speech engine.

Creates a fresh pyttsx3 instance per utterance inside a daemon thread
to avoid the macOS NSSpeechSynthesizer hang bug that occurs when
reusing an engine across threads.
"""

import threading
import platform
import subprocess


class TTSEngine:
    """Lightweight TTS wrapper safe for use alongside Gradio's event loop."""

    def __init__(self, rate=150, volume=0.9):
        self._rate = rate
        self._volume = volume
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API (matches the interface Person 4 expects)
    # ------------------------------------------------------------------

    def speak(self, text):
        """Speak *text* without blocking the caller."""
        if not text:
            return
        thread = threading.Thread(
            target=self._speak_sync,
            args=(text,),
            daemon=True,
        )
        thread.start()

    def stop(self):
        """No-op for interface compatibility."""
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _speak_sync(self, text):
        with self._lock:  # prevent overlapping utterances
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                # Fallback: macOS built-in `say` command
                if platform.system() == "Darwin":
                    try:
                        subprocess.run(
                            ["say", text],
                            timeout=15,
                            check=False,
                        )
                    except Exception:
                        pass
