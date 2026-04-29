"""Sanity checks for feature engineering.

We don't need MediaPipe here — we synthesise plausible 21x3 landmark arrays
to verify invariances (translation, scale, mirror).
"""
import numpy as np

from src.gestures.features import (
    FEATURE_DIM, landmarks_to_features,
)
from src.gestures.landmarks import HandLandmarks


def _make_hand(coords: np.ndarray, handed: str = "Right") -> HandLandmarks:
    return HandLandmarks(coords=coords.astype(np.float32), handedness=handed, score=0.99)


def _fake_open_palm() -> np.ndarray:
    """A toy 21x3 layout vaguely resembling an open palm."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.3, 0.7, size=(21, 3)).astype(np.float32)
    coords[0] = [0.5, 0.9, 0.0]      # wrist near bottom
    coords[9] = [0.5, 0.5, 0.0]      # middle MCP
    coords[12] = [0.5, 0.1, 0.0]     # middle tip up
    return coords


def test_feature_dim():
    feats = landmarks_to_features(_make_hand(_fake_open_palm()))
    assert feats.shape == (FEATURE_DIM,)


def test_translation_invariance():
    coords = _fake_open_palm()
    f1 = landmarks_to_features(_make_hand(coords))
    coords2 = coords + np.array([0.1, -0.05, 0.0], dtype=np.float32)
    f2 = landmarks_to_features(_make_hand(coords2))
    np.testing.assert_allclose(f1, f2, atol=1e-5)


def test_scale_invariance():
    coords = _fake_open_palm() - 0.5  # center near origin so scaling is meaningful
    coords += 0.5
    f1 = landmarks_to_features(_make_hand(coords))
    coords2 = (coords - 0.5) * 2.0 + 0.5
    f2 = landmarks_to_features(_make_hand(coords2))
    # Scale invariance is approximate (z is not rescaled by image), so allow slack.
    np.testing.assert_allclose(f1, f2, atol=1e-4)


def test_mirror_left_to_right():
    coords = _fake_open_palm()
    fr = landmarks_to_features(_make_hand(coords, "Right"))

    mirrored = coords.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    fl = landmarks_to_features(_make_hand(mirrored, "Left"))
    np.testing.assert_allclose(fr, fl, atol=1e-5)
