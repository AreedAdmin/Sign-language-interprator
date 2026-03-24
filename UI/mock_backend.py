"""
Mock backend for UI development.

Simulates hand detection, sign classification, and Claude sentence refinement
so the UI can be built and demoed independently of the real backend.
"""

import time
import math
import random
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Sign vocabulary (matches the 20-word WLASL target list)
# ---------------------------------------------------------------------------

SIGNS = [
    "hello", "thank you", "please", "sorry", "friend",
    "help", "good", "bad", "happy", "stop",
    "yes", "no", "want", "water", "more",
    "learn", "food", "love", "peace", "name",
]

# ---------------------------------------------------------------------------
# Pre-written Claude-style sentence refinements
# ---------------------------------------------------------------------------

_REFINEMENTS = {
    ("hello",): "Hello!",
    ("hello", "friend"): "Hello, friend!",
    ("hello", "friend", "how"): "Hello friend, how are you?",
    ("thank you",): "Thank you!",
    ("please", "help"): "Please help me.",
    ("want", "water"): "I want some water.",
    ("want", "food"): "I want some food.",
    ("yes", "please"): "Yes, please!",
    ("no", "thank you"): "No, thank you.",
    ("good", "friend"): "Good friend.",
    ("hello", "name"): "Hello, what is your name?",
    ("help", "please"): "Help me, please!",
    ("love", "friend"): "I love my friend.",
    ("more", "water"): "More water, please.",
    ("more", "food"): "More food, please.",
    ("happy",): "I'm happy!",
    ("sorry",): "I'm sorry.",
    ("stop",): "Stop!",
    ("learn", "more"): "I want to learn more.",
}

# ---------------------------------------------------------------------------
# MediaPipe-like hand landmark connections (for drawing skeleton)
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

# A template of 21 normalised hand landmarks (roughly a right hand)
_HAND_TEMPLATE = np.array([
    [0.50, 0.80, 0.0],  # 0  wrist
    [0.42, 0.72, 0.0],  # 1  thumb_cmc
    [0.36, 0.64, 0.0],  # 2  thumb_mcp
    [0.32, 0.56, 0.0],  # 3  thumb_ip
    [0.28, 0.50, 0.0],  # 4  thumb_tip
    [0.42, 0.52, 0.0],  # 5  index_mcp
    [0.40, 0.40, 0.0],  # 6  index_pip
    [0.39, 0.32, 0.0],  # 7  index_dip
    [0.38, 0.24, 0.0],  # 8  index_tip
    [0.50, 0.50, 0.0],  # 9  middle_mcp
    [0.50, 0.38, 0.0],  # 10 middle_pip
    [0.50, 0.30, 0.0],  # 11 middle_dip
    [0.50, 0.22, 0.0],  # 12 middle_tip
    [0.57, 0.52, 0.0],  # 13 ring_mcp
    [0.58, 0.40, 0.0],  # 14 ring_pip
    [0.59, 0.32, 0.0],  # 15 ring_dip
    [0.60, 0.26, 0.0],  # 16 ring_tip
    [0.64, 0.56, 0.0],  # 17 pinky_mcp
    [0.66, 0.46, 0.0],  # 18 pinky_pip
    [0.67, 0.40, 0.0],  # 19 pinky_dip
    [0.68, 0.34, 0.0],  # 20 pinky_tip
], dtype=np.float32)


def _jitter_landmarks(base, amount=0.008):
    """Add small random jitter to landmarks for realism."""
    noise = np.random.uniform(-amount, amount, base.shape).astype(np.float32)
    return base + noise


# ---------------------------------------------------------------------------
# Mock functions
# ---------------------------------------------------------------------------

def mock_detect_hand(frame):
    """
    Simulate hand detection.

    Draws landmarks on the frame and returns detection info.
    Detection cycles on/off every few seconds to feel realistic.
    """
    if frame is None:
        return {"hand_detected": False, "landmarks": [], "confidence": 0.0,
                "annotated_frame": frame}

    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Simulate detection cycling: detected for 4s, gone for 1.5s
    cycle = time.time() % 5.5
    hand_detected = cycle < 4.0

    if hand_detected:
        landmarks = _jitter_landmarks(_HAND_TEMPLATE)
        confidence = round(random.uniform(0.82, 0.98), 2)

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
            pt2 = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
            cv2.line(annotated, pt1, pt2, (0, 255, 170), 2, cv2.LINE_AA)

        # Draw landmark dots
        for lm in landmarks:
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(annotated, (cx, cy), 5, (0, 212, 170), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

        return {
            "hand_detected": True,
            "landmarks": landmarks.tolist(),
            "confidence": confidence,
            "annotated_frame": annotated,
        }

    return {
        "hand_detected": False,
        "landmarks": [],
        "confidence": 0.0,
        "annotated_frame": annotated,
    }


def mock_classify_sign():
    """
    Simulate sign classification — rotates through SIGNS every ~2.5 s.

    Returns dict matching the real classifier contract.
    """
    idx = int(time.time() / 2.5) % len(SIGNS)
    sign = SIGNS[idx]
    conf = round(random.uniform(0.78, 0.96), 2)

    # Build plausible top-3
    others = [s for s in SIGNS if s != sign]
    second = random.choice(others)
    others2 = [s for s in others if s != second]
    third = random.choice(others2)

    top_3 = [
        (sign, conf),
        (second, round(random.uniform(0.02, 0.10), 2)),
        (third, round(random.uniform(0.01, 0.05), 2)),
    ]

    return {"sign": sign, "confidence": conf, "top_3": top_3}


def mock_claude_refine(raw_signs):
    """
    Simulate Claude sentence refinement.

    Checks a lookup table first; falls back to simple join.
    """
    if not raw_signs:
        return {"raw_signs": [], "refined_sentence": ""}

    key = tuple(raw_signs)

    # Try exact match, then progressively shorter suffixes
    for start in range(len(key)):
        sub = key[start:]
        if sub in _REFINEMENTS:
            return {"raw_signs": list(raw_signs), "refined_sentence": _REFINEMENTS[sub]}

    # Fallback: capitalise and join
    sentence = " ".join(raw_signs).capitalize()
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return {"raw_signs": list(raw_signs), "refined_sentence": sentence}
