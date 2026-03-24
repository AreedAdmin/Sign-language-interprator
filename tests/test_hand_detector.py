"""
tests/test_hand_detector.py — Person 2 (Fares)
Unit tests for hand_detector.py, preprocessing.py, and feature_extraction.py.
Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from utils.preprocessing import (
    resize_frame,
    flip_frame,
    normalise_landmarks_to_wrist,
    landmarks_to_bounding_box,
    is_hand_stable,
)
from utils.feature_extraction import (
    extract_feature_vector,
    extract_finger_angles,
    extract_fingertip_distances,
    extract_combined_features,
    fingers_extended,
    is_fist,
    is_open_hand,
    is_thumbs_up,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_frame():
    """A random 480×640 BGR frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def flat_landmarks():
    """21 landmarks all at (0.5, 0.5, 0.0) — degenerate but valid for shape tests."""
    return [(0.5, 0.5, 0.0)] * 21


@pytest.fixture
def spread_landmarks():
    """21 landmarks spread across the frame — more realistic."""
    rng = np.random.default_rng(42)
    coords = rng.uniform(0.1, 0.9, (21, 3)).tolist()
    return [tuple(c) for c in coords]


# ─────────────────────────────────────────────────────────────
# preprocessing.py
# ─────────────────────────────────────────────────────────────

class TestResizeFrame:
    def test_output_shape(self, dummy_frame):
        out = resize_frame(dummy_frame, 320, 240)
        assert out.shape == (240, 320, 3)

    def test_default_shape(self, dummy_frame):
        out = resize_frame(dummy_frame)
        assert out.shape == (480, 640, 3)

    def test_dtype_preserved(self, dummy_frame):
        out = resize_frame(dummy_frame, 320, 240)
        assert out.dtype == np.uint8


class TestFlipFrame:
    def test_horizontal_flip_shape(self, dummy_frame):
        out = flip_frame(dummy_frame, flip_code=1)
        assert out.shape == dummy_frame.shape

    def test_flip_is_different(self, dummy_frame):
        out = flip_frame(dummy_frame, flip_code=1)
        assert not np.array_equal(out, dummy_frame)

    def test_double_flip_roundtrip(self, dummy_frame):
        once  = flip_frame(dummy_frame, flip_code=1)
        twice = flip_frame(once, flip_code=1)
        assert np.array_equal(twice, dummy_frame)


class TestNormaliseLandmarks:
    def test_output_length(self, spread_landmarks):
        flat = normalise_landmarks_to_wrist(spread_landmarks)
        assert len(flat) == 63

    def test_first_three_near_zero(self, spread_landmarks):
        """Wrist should map to origin after normalisation."""
        flat = normalise_landmarks_to_wrist(spread_landmarks)
        assert abs(flat[0]) < 1e-6
        assert abs(flat[1]) < 1e-6
        assert abs(flat[2]) < 1e-6

    def test_empty_returns_zeros(self):
        flat = normalise_landmarks_to_wrist([])
        assert flat == [0.0] * 63

    def test_wrong_count_returns_zeros(self):
        flat = normalise_landmarks_to_wrist([(0.5, 0.5, 0.0)] * 10)
        assert flat == [0.0] * 63


class TestBoundingBox:
    def test_returns_four_ints(self, spread_landmarks):
        bbox = landmarks_to_bounding_box(spread_landmarks, 640, 480)
        assert len(bbox) == 4
        assert all(isinstance(v, int) for v in bbox)

    def test_bbox_within_frame(self, spread_landmarks):
        x, y, w, h = landmarks_to_bounding_box(spread_landmarks, 640, 480, padding=0)
        assert x >= 0 and y >= 0
        assert x + w <= 640
        assert y + h <= 480


class TestHandStability:
    def test_stable_hand_detected(self, spread_landmarks):
        """Identical landmarks across frames → stable."""
        buf = [spread_landmarks] * 5
        assert is_hand_stable(buf, threshold=0.05) is True

    def test_moving_hand_not_stable(self):
        """Landmarks that jump around → not stable."""
        rng = np.random.default_rng(0)
        buf = [
            [tuple(rng.uniform(0, 1, 3)) for _ in range(21)]
            for _ in range(5)
        ]
        assert is_hand_stable(buf, threshold=0.001) is False

    def test_none_entries_handled(self, spread_landmarks):
        buf = [None, spread_landmarks, None, spread_landmarks]
        # Only 2 valid frames — should not crash
        result = is_hand_stable(buf)
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────
# feature_extraction.py
# ─────────────────────────────────────────────────────────────

class TestExtractFeatureVector:
    def test_output_shape(self, spread_landmarks):
        fv = extract_feature_vector(spread_landmarks)
        assert fv.shape == (63,)

    def test_dtype(self, spread_landmarks):
        fv = extract_feature_vector(spread_landmarks)
        assert fv.dtype == np.float32

    def test_empty_input_returns_zeros(self):
        fv = extract_feature_vector([])
        assert np.all(fv == 0)

    def test_first_three_near_zero(self, spread_landmarks):
        """Wrist (index 0) should be at origin."""
        fv = extract_feature_vector(spread_landmarks)
        assert abs(fv[0]) < 1e-5
        assert abs(fv[1]) < 1e-5
        assert abs(fv[2]) < 1e-5


class TestFingerAngles:
    def test_output_shape(self, spread_landmarks):
        fa = extract_finger_angles(spread_landmarks)
        assert fa.shape == (5,)

    def test_values_in_range(self, spread_landmarks):
        fa = extract_finger_angles(spread_landmarks)
        assert np.all(fa >= -1.0) and np.all(fa <= 1.0)


class TestFingertipDistances:
    def test_output_shape(self, spread_landmarks):
        fd = extract_fingertip_distances(spread_landmarks)
        assert fd.shape == (5,)

    def test_non_negative(self, spread_landmarks):
        fd = extract_fingertip_distances(spread_landmarks)
        assert np.all(fd >= 0)


class TestCombinedFeatures:
    def test_output_shape(self, spread_landmarks):
        cf = extract_combined_features(spread_landmarks)
        assert cf.shape == (73,)


class TestGestureHelpers:
    def test_fingers_extended_returns_list_of_5(self, spread_landmarks):
        ext = fingers_extended(spread_landmarks)
        assert len(ext) == 5
        assert all(isinstance(v, bool) for v in ext)

    def test_is_fist_does_not_crash(self, spread_landmarks):
        result = is_fist(spread_landmarks)
        assert isinstance(result, bool)

    def test_is_open_hand_does_not_crash(self, spread_landmarks):
        result = is_open_hand(spread_landmarks)
        assert isinstance(result, bool)

    def test_is_thumbs_up_does_not_crash(self, spread_landmarks):
        result = is_thumbs_up(spread_landmarks)
        assert isinstance(result, bool)

    def test_empty_input_is_fist_false(self):
        assert is_fist([]) is False

    def test_empty_input_is_open_hand_false(self):
        assert is_open_hand([]) is False


# ─────────────────────────────────────────────────────────────
# HandDetector integration (no webcam — uses a black frame)
# ─────────────────────────────────────────────────────────────

class TestHandDetectorWithBlackFrame:
    """
    These tests import HandDetector and run it on a synthetic black frame
    (no real hand present).  We verify the output contract, not accuracy.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from models.hand_detector import HandDetector
        self.detector = HandDetector()
        self.black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        yield
        self.detector.close()

    def test_detect_returns_dict(self):
        result = self.detector.detect(self.black_frame)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = self.detector.detect(self.black_frame)
        expected_keys = {
            "hand_present", "landmarks", "smoothed_landmarks",
            "temporal_landmarks", "annotated_frame", "confidence",
            "timestamp", "frame_id",
        }
        assert expected_keys.issubset(result.keys())

    def test_no_hand_on_black_frame(self):
        result = self.detector.detect(self.black_frame)
        assert result["hand_present"] is False
        assert result["landmarks"] is None

    def test_annotated_frame_same_shape(self):
        result = self.detector.detect(self.black_frame)
        assert result["annotated_frame"].shape == self.black_frame.shape

    def test_frame_id_increments(self):
        r1 = self.detector.detect(self.black_frame)
        r2 = self.detector.detect(self.black_frame)
        assert r2["frame_id"] == r1["frame_id"] + 1

    def test_statistics_structure(self):
        self.detector.detect(self.black_frame)
        stats = self.detector.get_statistics()
        assert "total_frames" in stats
        assert "approx_fps" in stats
        assert stats["total_frames"] >= 1
