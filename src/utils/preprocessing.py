"""
preprocessing.py — Person 2 (Fares) — Computer Vision utilities
Frame-level preprocessing helpers: resizing, flipping, cropping,
lighting normalisation, and landmark coordinate normalisation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# Frame preprocessing
# ─────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Resize a BGR frame to a fixed resolution."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def flip_frame(frame: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Mirror the frame (default: horizontal flip so webcam acts like a mirror).
    flip_code: 1 = horizontal, 0 = vertical, -1 = both.
    """
    return cv2.flip(frame, flip_code)


def normalise_lighting(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the
    luminance channel of the frame.  Helps in dim / uneven lighting.
    """
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def crop_roi(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop a rectangular region-of-interest from a frame."""
    return frame[y : y + h, x : x + w]


def preprocess_frame(
    frame: np.ndarray,
    target_width: int = 640,
    target_height: int = 480,
    mirror: bool = True,
    enhance_lighting: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single webcam frame.

    Args:
        frame:            Raw BGR frame from cv2.VideoCapture.
        target_width/height: Output resolution.
        mirror:           Flip horizontally (natural mirror view).
        enhance_lighting: Apply CLAHE — useful in dark environments.

    Returns:
        Preprocessed BGR frame ready for MediaPipe.
    """
    out = resize_frame(frame, target_width, target_height)
    if mirror:
        out = flip_frame(out)
    if enhance_lighting:
        out = normalise_lighting(out)
    return out


# ─────────────────────────────────────────────────────────────
# Landmark coordinate normalisation
# ─────────────────────────────────────────────────────────────

LandmarkList = List[Tuple[float, float, float]]


def normalise_landmarks_to_wrist(
    landmarks: LandmarkList,
) -> List[float]:
    """
    Translate all landmarks so the wrist (index 0) is the origin,
    then scale so the longest distance from the wrist = 1.
    Returns a flat list of 63 floats (21 × x,y,z).

    This is the format expected by Person 1's ASLClassifier.
    """
    if not landmarks or len(landmarks) != 21:
        return [0.0] * 63

    wrist = landmarks[0]
    translated = [(x - wrist[0], y - wrist[1], z - wrist[2])
                  for x, y, z in landmarks]

    # Scale factor: max Euclidean distance from wrist
    distances = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in translated]
    max_dist  = max(distances) if max(distances) > 1e-6 else 1.0

    flat = []
    for x, y, z in translated:
        flat.extend([x / max_dist, y / max_dist, z / max_dist])

    return flat


def landmarks_to_bounding_box(
    landmarks: LandmarkList,
    frame_width: int,
    frame_height: int,
    padding: int = 20,
) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box (x, y, w, h) around all 21 landmarks.
    Useful for cropping the hand region before further analysis.
    """
    xs = [lm[0] * frame_width  for lm in landmarks]
    ys = [lm[1] * frame_height for lm in landmarks]

    x_min = max(0, int(min(xs)) - padding)
    y_min = max(0, int(min(ys)) - padding)
    x_max = min(frame_width,  int(max(xs)) + padding)
    y_max = min(frame_height, int(max(ys)) + padding)

    return x_min, y_min, x_max - x_min, y_max - y_min


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 200, 80),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box with an optional label on the frame (in-place)."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(
            frame, label, (x, max(y - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return frame


# ─────────────────────────────────────────────────────────────
# Gesture segmentation helpers
# ─────────────────────────────────────────────────────────────

def is_hand_stable(
    landmark_buffer: List[Optional[LandmarkList]],
    threshold: float = 0.02,
) -> bool:
    """
    Detect whether the hand has been roughly still for the last N frames.
    Useful for triggering a classification only when the hand is held steady.

    Args:
        landmark_buffer: Last N frames of landmarks (None = no detection).
        threshold:        Max allowed average displacement between frames (normalised units).

    Returns:
        True if the hand appears to be steady.
    """
    valid = [lm for lm in landmark_buffer if lm is not None]
    if len(valid) < 2:
        return False

    total_motion = 0.0
    count = 0

    for i in range(1, len(valid)):
        for j in range(21):
            dx = valid[i][j][0] - valid[i-1][j][0]
            dy = valid[i][j][1] - valid[i-1][j][1]
            total_motion += np.sqrt(dx**2 + dy**2)
            count += 1

    avg_motion = total_motion / count if count > 0 else 0.0
    return avg_motion < threshold
