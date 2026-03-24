"""
ASL Real-Time Interpreter — Gradio UI
Person 3: UI/Frontend Developer

Run standalone (mock mode):   python3 app.py
Run with real backend:        python3 app.py --live

Integration (Person 4):
    from app import ASLGradioApp
    app = ASLGradioApp()
    app.set_pipeline(real_pipeline_func)
    app.tts_engine = real_tts_engine
    demo = app.create_interface()
    demo.launch(css=_CSS)
"""

import os
import argparse
import gradio as gr
import time

try:
    from .tts_engine import TTSEngine          # package import (src/ui)
except ImportError:
    from tts_engine import TTSEngine           # direct run fallback

# Load CSS from file
_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles.css")
try:
    with open(_CSS_PATH) as f:
        _CSS = f.read()
except FileNotFoundError:
    _CSS = ""


# ═══════════════════════════════════════════════════════════════════════════
# HTML builders (pure functions — no state)
# ═══════════════════════════════════════════════════════════════════════════

def _detection_html(sign, confidence, top_3):
    """Rich HTML for the detection panel."""
    if confidence >= 0.80:
        bar_color = "#00d4aa"
    elif confidence >= 0.60:
        bar_color = "#ffd93d"
    else:
        bar_color = "#ff6b6b"

    pct = int(confidence * 100)

    alternatives = " &middot; ".join(
        f"{s} ({int(c * 100)}%)" for s, c in top_3[1:]
    )

    return f"""
    <div style="text-align:center; padding:24px 12px;">
        <div style="font-size:3.5rem; font-weight:800; color:{bar_color};
                    text-shadow:0 0 25px {bar_color}44; margin-bottom:14px;
                    letter-spacing:1px;">
            {sign.upper()}
        </div>
        <div style="background:#0f0f23; border-radius:8px; height:30px;
                    overflow:hidden; margin:0 auto 12px; max-width:280px;
                    border:1px solid rgba(255,255,255,0.06);">
            <div style="width:{pct}%; height:100%;
                        background:linear-gradient(90deg, #00d4aa, #7c5cfc);
                        border-radius:8px; display:flex; align-items:center;
                        justify-content:center; color:#fff; font-weight:600;
                        font-size:0.85rem; transition:width 0.3s ease;">
                {pct}%
            </div>
        </div>
        <div style="color:#555; font-size:0.85rem;">
            Also: {alternatives}
        </div>
    </div>
    """


def _waiting_html():
    """HTML shown when no hand is detected."""
    return """
    <div style="text-align:center; padding:48px 20px; color:#444;">
        <div style="font-size:3.5rem; margin-bottom:10px; opacity:0.5;">🤚</div>
        <div style="font-size:1.05rem;">Waiting for hand&hellip;</div>
    </div>
    """


def _sentence_html(signs, refined):
    """HTML for the sentence builder + Claude refinement section."""
    if not signs:
        return """
        <div style="text-align:center; padding:18px; color:#444; font-size:0.95rem;">
            Detected signs will appear here&hellip;
        </div>
        """

    chips = []
    for i, s in enumerate(signs):
        is_last = i == len(signs) - 1
        if is_last:
            chips.append(
                f'<span style="display:inline-block; background:#00d4aa22; '
                f'border:1px solid #00d4aa; border-radius:20px; padding:5px 14px; '
                f'margin:3px 4px; font-weight:600; color:#00d4aa; '
                f'box-shadow:0 0 8px #00d4aa33;">{s}</span>'
            )
        else:
            chips.append(
                f'<span style="display:inline-block; background:#7c5cfc22; '
                f'border:1px solid #7c5cfc55; border-radius:20px; padding:5px 14px; '
                f'margin:3px 4px; color:#b8a9ff;">{s}</span>'
            )

    chips_html = " ".join(chips)

    claude_html = ""
    if refined:
        claude_html = f"""
        <div style="margin-top:14px; padding:12px 16px; background:#16213e;
                    border-left:3px solid #7c5cfc; border-radius:0 8px 8px 0;">
            <div style="font-size:0.75rem; color:#7c5cfc; font-weight:600;
                        margin-bottom:4px; letter-spacing:0.5px;">CLAUDE</div>
            <div style="color:#e8e8e8; font-size:1.05rem;">
                &ldquo;{refined}&rdquo;
            </div>
        </div>
        """

    return f"""
    <div style="padding:10px 4px;">
        <div style="font-size:0.8rem; color:#666; margin-bottom:8px; font-weight:600;
                    letter-spacing:0.3px;">SIGNS DETECTED</div>
        <div style="line-height:2.2;">
            {chips_html}
        </div>
        {claude_html}
    </div>
    """


def _history_html(history):
    """HTML for the inline sign history (last 5)."""
    if not history:
        return '<div style="color:#444; font-size:0.85rem; padding:8px;">No signs yet.</div>'

    rows = []
    for entry in reversed(history[-5:]):
        conf = int(entry["confidence"] * 100)
        if conf >= 80:
            c = "#00d4aa"
        elif conf >= 60:
            c = "#ffd93d"
        else:
            c = "#ff6b6b"

        rows.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<td style="padding:4px 6px; color:#555; font-size:0.78rem;">{entry["time"]}</td>'
            f'<td style="padding:4px 6px; color:#e8e8e8; font-weight:600; font-size:0.9rem;">{entry["sign"]}</td>'
            f'<td style="padding:4px 6px;"><span style="color:{c}; font-weight:600; '
            f'font-size:0.82rem;">{conf}%</span></td></tr>'
        )

    th_style = (
        'padding:3px 6px; text-align:left; color:#555; font-size:0.7rem; '
        'font-weight:600; letter-spacing:0.3px;'
    )
    return (
        f'<table style="width:100%; border-collapse:collapse;">'
        f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.08);">'
        f'<th style="{th_style}">TIME</th>'
        f'<th style="{th_style}">SIGN</th>'
        f'<th style="{th_style}">CONF</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _stats_html(sign_count, session_start):
    """HTML for the live stats bar."""
    elapsed = int(time.time() - session_start)
    mins, secs = divmod(elapsed, 60)
    return f"""
    <div style="display:flex; justify-content:center; gap:28px; padding:8px 0;
                font-size:0.85rem; color:#555;">
        <span><strong style="color:#00d4aa;">{sign_count}</strong> signs detected</span>
        <span>session: {mins}m {secs:02d}s</span>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════
# ASLGradioApp — the class Person 4 integrates with
# ═══════════════════════════════════════════════════════════════════════════

class ASLGradioApp:
    """
    Gradio-based UI for the ASL interpreter.

    Integration API (for Person 4 / main.py):
        app = ASLGradioApp()
        app.set_pipeline(pipeline_func)   # frame → dict
        app.tts_engine = TTSEngine(...)   # must have .speak(text)
        demo = app.create_interface()
        demo.launch(css=_CSS)
    """

    def __init__(self, use_mock=True):
        self._use_mock = use_mock
        self._pipeline = None

        self.tts_engine = TTSEngine(rate=150, volume=0.9)

        # Session state
        self._last_sign_time = 0.0
        self._cooldown = 1.8
        self._accumulated_signs = []
        self._refined_sentence = ""
        self._last_detected_sign = None
        self._sign_history = []
        self._session_start = time.time()

    # ── Public integration API ─────────────────────────────────

    def set_pipeline(self, pipeline_func):
        """
        Set the real processing pipeline.

        pipeline_func(frame) → {
            "frame": np.ndarray,
            "prediction": {"sign": str, "confidence": float, "top_3": [...]} | None,
            "landmarks_detected": bool,
        }
        """
        self._pipeline = pipeline_func
        self._use_mock = False

    # ── Internal processing ────────────────────────────────────

    def _process_frame(self, frame):
        """Central frame callback — returns 5 outputs."""
        empty = (
            None,
            _waiting_html(),
            _sentence_html(self._accumulated_signs, self._refined_sentence),
            _history_html(self._sign_history),
            _stats_html(len(self._sign_history), self._session_start),
        )
        if frame is None:
            return empty

        try:
            if self._use_mock:
                return self._process_mock(frame)
            else:
                return self._process_live(frame)
        except Exception as e:
            error_html = f"""
            <div style="text-align:center; padding:24px; color:#ff6b6b;">
                <div style="font-size:1.4rem; margin-bottom:8px;">Detection paused</div>
                <div style="font-size:0.85rem; color:#666;">{e}</div>
            </div>
            """
            return (
                frame, error_html,
                _sentence_html(self._accumulated_signs, self._refined_sentence),
                _history_html(self._sign_history),
                _stats_html(len(self._sign_history), self._session_start),
            )

    def _process_mock(self, frame):
        try:
            from .mock_backend import mock_detect_hand, mock_classify_sign, mock_claude_refine
        except ImportError:
            from mock_backend import mock_detect_hand, mock_classify_sign, mock_claude_refine

        detection = mock_detect_hand(frame)
        annotated = detection["annotated_frame"]

        if detection["hand_detected"]:
            prediction = mock_classify_sign()
            det_html = _detection_html(
                prediction["sign"], prediction["confidence"], prediction["top_3"],
            )
            self._maybe_accumulate(
                prediction["sign"], prediction["confidence"],
                refine_fn=mock_claude_refine,
            )
        else:
            det_html = _waiting_html()

        return (
            annotated, det_html,
            _sentence_html(self._accumulated_signs, self._refined_sentence),
            _history_html(self._sign_history),
            _stats_html(len(self._sign_history), self._session_start),
        )

    def _process_live(self, frame):
        result = self._pipeline(frame)
        annotated = result.get("frame", frame)
        prediction = result.get("prediction")

        if prediction and prediction.get("sign"):
            top_3 = prediction.get("top_3", [
                (prediction["sign"], prediction["confidence"]),
                ("—", 0), ("—", 0),
            ])
            det_html = _detection_html(
                prediction["sign"], prediction["confidence"], top_3,
            )
            self._maybe_accumulate(
                prediction["sign"], prediction["confidence"],
            )
        else:
            det_html = _waiting_html()

        return (
            annotated, det_html,
            _sentence_html(self._accumulated_signs, self._refined_sentence),
            _history_html(self._sign_history),
            _stats_html(len(self._sign_history), self._session_start),
        )

    def _maybe_accumulate(self, sign, confidence, refine_fn=None):
        now = time.time()
        if (
            confidence >= 0.70
            and sign != self._last_detected_sign
            and (now - self._last_sign_time) >= self._cooldown
        ):
            self._accumulated_signs.append(sign)
            self._last_detected_sign = sign
            self._last_sign_time = now

            self._sign_history.append({
                "sign": sign,
                "confidence": confidence,
                "time": time.strftime("%H:%M:%S"),
            })

            if refine_fn:
                result = refine_fn(self._accumulated_signs)
                self._refined_sentence = result["refined_sentence"]
            else:
                self._refined_sentence = " ".join(self._accumulated_signs).capitalize() + "."

    def _clear_sentence(self):
        self._accumulated_signs = []
        self._refined_sentence = ""
        self._last_detected_sign = None
        self._last_sign_time = 0.0
        return _sentence_html([], "")

    def _speak_sentence(self):
        if self._refined_sentence:
            self.tts_engine.speak(self._refined_sentence)
            return f"""
            <div style="text-align:center; padding:8px; color:#00d4aa; font-size:0.9rem;">
                🔊 Speaking: &ldquo;{self._refined_sentence}&rdquo;
            </div>
            """
        return """
        <div style="text-align:center; padding:8px; color:#555; font-size:0.9rem;">
            Nothing to speak yet.
        </div>
        """

    # ── Gradio interface builder ───────────────────────────────

    def create_interface(self):
        mode_label = "Mock" if self._use_mock else "Live"

        with gr.Blocks(title="ASL Real-Time Interpreter") as demo:

            # ── Gradient header banner ──
            gr.HTML(f"""
            <div style="background:linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                        border:1px solid rgba(0,212,170,0.15); border-radius:12px;
                        padding:22px 16px; text-align:center;
                        box-shadow:0 0 30px rgba(0,212,170,0.05); margin-bottom:4px;">
                <h1 style="margin:0; font-size:2.2rem; color:#e8e8e8; font-weight:700;">
                    🤟 ASL Real-Time Interpreter
                </h1>
                <p style="margin:6px 0 0; color:#555; font-size:0.95rem;">
                    <span class="live-dot"></span>
                    {mode_label} &mdash; real-time sign language detection &amp; translation
                </p>
            </div>
            """)

            # ── Live stats bar ──
            stats_bar = gr.HTML(
                value=_stats_html(0, self._session_start),
                elem_id="stats-bar",
            )

            # ── Main two-column layout ──
            with gr.Row():
                # Left — webcam + processed feed
                with gr.Column(scale=3):
                    webcam = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                        label="Camera",
                        elem_id="webcam-input",
                        height=80,
                    )
                    output_img = gr.Image(
                        type="numpy",
                        label="Live Feed",
                        elem_id="hero-video",
                    )

                # Right — detection + recent history
                with gr.Column(scale=2):
                    gr.HTML("""
                    <div style="font-weight:600; font-size:1rem; color:#e8e8e8;
                                padding:0 0 4px; letter-spacing:0.3px;">
                        DETECTION RESULTS
                    </div>
                    """)
                    detection_panel = gr.HTML(value=_waiting_html())

                    gr.HTML("""
                    <div style="border-top:1px solid rgba(255,255,255,0.06);
                                margin:8px 0; padding-top:10px;
                                font-weight:600; font-size:0.85rem; color:#666;
                                letter-spacing:0.3px;">
                        RECENT SIGNS
                    </div>
                    """)
                    history_panel = gr.HTML(value=_history_html([]))

            # ── Sentence builder (full width) ──
            gr.HTML("""
            <div style="padding:12px 0 4px; font-weight:600; font-size:1rem;
                        color:#e8e8e8; letter-spacing:0.3px;">
                SENTENCE BUILDER
            </div>
            """)
            sentence_panel = gr.HTML(value=_sentence_html([], ""))

            # ── Controls ──
            with gr.Row():
                speak_btn = gr.Button(
                    "🔊 Speak", variant="primary", elem_id="speak-btn", scale=2,
                )
                clear_btn = gr.Button(
                    "🗑 Clear", variant="stop", elem_id="clear-btn", scale=1,
                )

            # ── Speak feedback (visible) ──
            speak_feedback = gr.HTML(value="", elem_id="speak-feedback")

            # ── Footer ──
            gr.HTML("""
            <div class="footer-bar">
                Powered by Claude &middot; Claude Builder Club &middot; Spring 2026 Hackathon
                <br/>
                <span style="font-size:0.75rem; color:#444;">
                    Breaking communication barriers with AI
                </span>
            </div>
            """)

            # ── Events ──
            webcam.stream(
                fn=self._process_frame,
                inputs=[webcam],
                outputs=[output_img, detection_panel, sentence_panel,
                         history_panel, stats_bar],
                stream_every=0.1,
                time_limit=300,
            )

            clear_btn.click(
                fn=self._clear_sentence,
                outputs=[sentence_panel],
            )

            speak_btn.click(
                fn=self._speak_sentence,
                outputs=[speak_feedback],
            )

        return demo


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Real-Time Interpreter UI")
    parser.add_argument("--live", action="store_true",
                        help="Run with real backend (not mock)")
    args = parser.parse_args()

    app = ASLGradioApp(use_mock=not args.live)
    demo = app.create_interface()
    demo.launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        css=_CSS,
    )
