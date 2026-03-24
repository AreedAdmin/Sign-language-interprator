"""
feature_extraction.py — Person 2 (Fares) — Computer Vision utilities
Higher-level landmark feature engineering consumed by Person 1's classifier.
"""

import numpy as np
from typing import List, Tuple, Optional

LandmarkList = List[Tuple[float, float, float]]

# MediaPipe finger tip and MCP (knuckle) indices
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_MCPS  = [2, 5, 9, 13, 17]  # thumb IP, other MCPs
FINGER_PIPS  = [3, 6, 10, 14, 18]

WRIST_IDX    = 0
THUMB_TIP    = 4
INDEX_TIP    = 8
MIDDLE_TIP   = 12


# ─────────────────────────────────────────────────────────────
# Core feature vector
# ─────────────────────────────────────────────────────────────

def extract_feature_vector(landmarks: LandmarkList) -> np.ndarray:
    """
    Convert 21 raw (x, y, z) landmarks into a 63-dim normalised feature vector.
    Wrist-centred + scale-normalised.  This is the PRIMARY input to Person 1's model.

    Returns: np.ndarray of shape (63,), dtype float32.
    """
    if not landmarks or len(landmarks) != 21:
        return np.zeros(63, dtype=np.float32)

    wrist = np.array(landmarks[WRIST_IDX])
    pts   = np.array(landmarks)           # (21, 3)

    # Translate to wrist origin
    pts_centered = pts - wrist

    # Scale normalisation: distance wrist → middle finger MCP
    scale = np.linalg.norm(pts_centered[9])
    if scale < 1e-6:
        scale = 1.0

    pts_normalised = pts_centered / scale
    return pts_normalised.flatten().astype(np.float32)


def extract_finger_angles(landmarks: LandmarkList) -> np.ndarray:
    """
    Compute 5 bend angles (one per finger) as the cosine similarity
    between the proximal and distal phalanx vectors.
    Returns np.ndarray of shape (5,) in range [-1, 1].
    0 = fully extended, approaching -1 = tightly curled.
    """
    if not landmarks or len(landmarks) != 21:
        return np.zeros(5, dtype=np.float32)

    pts    = np.array(landmarks)
    angles = []

    # Finger segments: (MCP, PIP, TIP) — excluding thumb for simplicity
    segments = [
        (2, 3, 4),   # thumb: IP → tip direction
        (5, 6, 8),   # index
        (9, 10, 12), # middle
        (13, 14, 16),# ring
        (17, 18, 20),# pinky
    ]

    for a_idx, b_idx, c_idx in segments:
        v1 = pts[b_idx] - pts[a_idx]
        v2 = pts[c_idx] - pts[b_idx]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(0.0)
        else:
            cos_angle = np.dot(v1, v2) / (n1 * n2)
            angles.append(float(np.clip(cos_angle, -1.0, 1.0)))

    return np.array(angles, dtype=np.float32)


def extract_fingertip_distances(landmarks: LandmarkList) -> np.ndarray:
    """
    Compute the distance from each fingertip to the wrist, normalised
    by hand size (wrist → middle MCP distance).
    Returns np.ndarray of shape (5,).
    Large values = fingers extended; small values = fingers curled.
    """
    if not landmarks or len(landmarks) != 21:
        return np.zeros(5, dtype=np.float32)

    pts       = np.array(landmarks)
    wrist     = pts[WRIST_IDX]
    hand_size = np.linalg.norm(pts[9] - wrist)  # middle MCP
    if hand_size < 1e-6:
        hand_size = 1.0

    distances = []
    for tip_idx in FINGER_TIPS:
        d = np.linalg.norm(pts[tip_idx] - wrist) / hand_size
        distances.append(float(d))

    return np.array(distances, dtype=np.float32)


def extract_combined_features(landmarks: LandmarkList) -> np.ndarray:
    """
    Concatenate all feature types into one 73-dim vector:
      63  (normalised xyz) + 5 (finger angles) + 5 (tip distances)
    This richer representation can improve rule-based classification
    accuracy without requiring a trained ML model.
    """
    fv = extract_feature_vector(landmarks)
    fa = extract_finger_angles(landmarks)
    fd = extract_fingertip_distances(landmarks)
    return np.concatenate([fv, fa, fd]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Simple geometric helpers used by rule-based classifier backup
# ─────────────────────────────────────────────────────────────

def fingers_extended(landmarks: LandmarkList, threshold: float = 1.1) -> List[bool]:
    """
    Return a 5-bool list indicating whether each finger is extended.
    A finger is considered extended if its tip is further from the wrist
    than threshold × the distance wrist → MCP.
    Order: [thumb, index, middle, ring, pinky].
    """
    if not landmarks or len(landmarks) != 21:
        return [False] * 5

    pts       = np.array(landmarks)
    wrist     = pts[WRIST_IDX]

    extended = []
    pairs = list(zip(FINGER_TIPS, FINGER_MCPS))
    for tip_idx, mcp_idx in pairs:
        d_tip  = np.linalg.norm(pts[tip_idx] - wrist)
        d_mcp  = np.linalg.norm(pts[mcp_idx] - wrist)
        extended.append(bool(d_tip > threshold * d_mcp))

    return extended


def is_fist(landmarks: LandmarkList) -> bool:
    """All four fingers (not thumb) curled inward."""
    ext = fingers_extended(landmarks)
    return not any(ext[1:])   # index, middle, ring, pinky all closed


def is_open_hand(landmarks: LandmarkList) -> bool:
    """All five fingers extended."""
    ext = fingers_extended(landmarks)
    return all(ext)


def is_thumbs_up(landmarks: LandmarkList) -> bool:
    """Thumb extended upward, all other fingers closed."""
    if not landmarks or len(landmarks) != 21:
        return False

    ext = fingers_extended(landmarks)
    # Thumb extended, rest closed
    if not (ext[0] and not any(ext[1:])):
        return False

    # Thumb tip should be above (smaller y) the wrist
    pts = np.array(landmarks)
    return bool(pts[THUMB_TIP][1] < pts[WRIST_IDX][1])


def is_pointing(landmarks: LandmarkList) -> bool:
    """Index finger extended, others closed."""
    ext = fingers_extended(landmarks)
    return ext[1] and not ext[2] and not ext[3] and not ext[4]


def is_peace_sign(landmarks: LandmarkList) -> bool:
    """Index and middle extended, others closed."""
    ext = fingers_extended(landmarks)
    return ext[1] and ext[2] and not ext[3] and not ext[4]
