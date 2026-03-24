"""
asl_sign_mapper.py — Rule-based ASL sign recognition.

Combines MediaPipe gesture labels with hand landmark geometry to
output real ASL meanings (not just hand pose descriptions).

Priority: specific landmark rules first, MediaPipe gesture fallback second.
"""

import numpy as np
from typing import List, Tuple, Optional

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [2, 5, 9, 13, 17]
WRIST_IDX = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

LandmarkList = List[Tuple[float, float, float]]

# MediaPipe gesture name → ASL meaning (fallback when no rule matches)
_GESTURE_TO_ASL = {
    "Open_Palm":   "Hello",
    "Closed_Fist": "Yes",
    "Thumb_Up":    "Good",
    "Thumb_Down":  "Bad",
    "Pointing_Up": "One",
    "Victory":     "Peace",
    "ILoveYou":    "I Love You",
}


def _fingers_extended(
    landmarks: LandmarkList, threshold: float = 1.1
) -> List[bool]:
    """
    [thumb, index, middle, ring, pinky] — True if extended.
    """
    pts = np.array(landmarks)
    wrist = pts[WRIST_IDX]
    result = []
    for tip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_MCPS):
        d_tip = np.linalg.norm(pts[tip_idx] - wrist)
        d_mcp = np.linalg.norm(pts[mcp_idx] - wrist)
        result.append(bool(d_tip > threshold * d_mcp))
    return result


def _thumb_direction(landmarks: LandmarkList) -> str:
    """Return 'up', 'down', or 'side' based on thumb tip vs wrist."""
    pts = np.array(landmarks)
    dy = pts[THUMB_TIP][1] - pts[WRIST_IDX][1]
    if dy < -0.05:
        return "up"
    elif dy > 0.05:
        return "down"
    return "side"


def _finger_pattern(ext: List[bool]) -> str:
    """Convert [T, I, M, R, P] booleans to a string like 'TIMRP' or '-I---'."""
    labels = "TIMRP"
    return "".join(l if e else "-" for l, e in zip(labels, ext))


class ASLSignMapper:
    """
    Maps hand landmarks + optional MediaPipe gesture to an ASL sign meaning.

    Usage:
        mapper = ASLSignMapper()
        asl_sign = mapper.classify(landmarks, mediapipe_gesture)
        # asl_sign = "Hello" | "Good" | "I Love You" | None
    """

    def classify(
        self,
        landmarks: Optional[LandmarkList],
        mediapipe_gesture: Optional[str] = None,
        mediapipe_confidence: float = 0.0,
    ) -> Optional[str]:
        """
        Return the ASL meaning as a string, or None if no sign is detected.
        """
        if not landmarks or len(landmarks) != 21:
            return None

        ext = _fingers_extended(landmarks)
        pattern = _finger_pattern(ext)
        thumb_dir = _thumb_direction(landmarks)

        # ── Rule-based ASL signs (most specific first) ───────────

        # I Love You: thumb + index + pinky extended, middle + ring closed
        if ext[0] and ext[1] and not ext[2] and not ext[3] and ext[4]:
            return "I Love You"

        # Hello / Stop: all five fingers extended (open hand)
        if all(ext):
            return "Hello"

        # Good / Like: thumb up, all others closed
        if ext[0] and not any(ext[1:]) and thumb_dir == "up":
            return "Good"

        # Bad / Dislike: thumb down, all others closed
        if ext[0] and not any(ext[1:]) and thumb_dir == "down":
            return "Bad"

        # Thank You: all fingers extended, hand moving away (approximate
        # as open hand with thumb slightly aside — use MediaPipe to
        # disambiguate from Hello)
        # Handled by MediaPipe fallback since it's the same pose as Hello

        # Yes: closed fist
        if not any(ext):
            return "Yes"

        # No: index + middle extended and close together, rest closed
        # (ASL "no" is actually index+middle+thumb snapping, but
        #  static approximation uses just index+middle)
        if ext[1] and ext[2] and not ext[3] and not ext[4] and not ext[0]:
            return "No"

        # Peace / Two: index + middle extended, thumb may be out
        if ext[1] and ext[2] and not ext[3] and not ext[4]:
            return "Peace"

        # One / Wait: only index finger extended
        if ext[1] and not ext[2] and not ext[3] and not ext[4]:
            return "One"

        # Three / W: index + middle + ring extended
        if ext[1] and ext[2] and ext[3] and not ext[4]:
            return "Three"

        # Four / B: all fingers except thumb extended
        if not ext[0] and ext[1] and ext[2] and ext[3] and ext[4]:
            return "Four"

        # Call Me / Phone: thumb + pinky extended, others closed
        if ext[0] and not ext[1] and not ext[2] and not ext[3] and ext[4]:
            return "Phone"

        # I (pronoun): only pinky extended
        if not ext[0] and not ext[1] and not ext[2] and not ext[3] and ext[4]:
            return "I"

        # Rock On: index + pinky extended, others closed
        if not ext[0] and ext[1] and not ext[2] and not ext[3] and ext[4]:
            return "Rock On"

        # ── MediaPipe gesture fallback ───────────────────────────
        if mediapipe_gesture and mediapipe_gesture in _GESTURE_TO_ASL:
            if mediapipe_confidence >= 0.5:
                return _GESTURE_TO_ASL[mediapipe_gesture]

        return None
