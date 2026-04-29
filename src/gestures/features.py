"""Feature engineering on top of MediaPipe hand landmarks.

We produce a fixed-length feature vector per detected hand that is invariant
to:

* **translation** — we re-center on the wrist (landmark 0)
* **scale** — we divide by the wrist→middle-MCP distance (a robust hand size)
* **handedness** — left-hand frames are horizontally mirrored so the classifier
  sees a single canonical right-hand pose

The vector is the 21 × 3 normalized landmark coordinates flattened (63 dims),
concatenated with 11 engineered features chosen to discriminate ``open palm``
from ``thumbs up`` from random hand poses:

* 5 fingertip-to-wrist distances (one per finger)
* 5 finger extension ratios (tip→MCP / MCP→wrist)
* 1 thumb-vs-index spread (angle between thumb and index direction)

Total: **74 features per frame**.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .landmarks import (
    FINGER_MCPS, FINGER_TIPS, HandLandmarks, MIDDLE_MCP, INDEX_MCP, THUMB_MCP, WRIST,
)


FEATURE_DIM = 21 * 3 + 5 + 5 + 1  # 74


def _normalize_landmarks(coords: np.ndarray, mirror: bool) -> np.ndarray:
    """Translate to wrist, divide by hand size, optionally mirror x."""
    if mirror:
        coords = coords.copy()
        coords[:, 0] = 1.0 - coords[:, 0]  # flip horizontally in image space

    centered = coords - coords[WRIST]
    hand_size = np.linalg.norm(centered[MIDDLE_MCP])
    if hand_size < 1e-6:
        return centered  # degenerate, return as-is
    return centered / hand_size


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle (in radians) between two 2D/3D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.arccos(cos))


def landmarks_to_features(
    hand: HandLandmarks,
    *,
    canonical_right: bool = True,
) -> np.ndarray:
    """Convert a :class:`HandLandmarks` into a fixed-length feature vector.

    Parameters
    ----------
    canonical_right : bool
        If True, mirror left hands so the classifier always sees a "right
        hand". This gives more training data per pose at the cost of losing
        handedness as a feature (we don't need it for our gestures).
    """
    mirror = canonical_right and hand.handedness == "Left"
    norm = _normalize_landmarks(hand.coords, mirror=mirror)

    # Flat 63-dim base features.
    base = norm.flatten()  # (63,)

    # 5 fingertip-to-wrist distances.
    tip_dists = np.linalg.norm(norm[list(FINGER_TIPS)], axis=1)  # (5,)

    # 5 extension ratios: how far the tip is from the wrist relative to the
    # MCP joint of the same finger. >1 means finger extended past MCP, near 1
    # or below means finger curled in.
    mcp_dists = np.linalg.norm(norm[list(FINGER_MCPS)], axis=1)
    extension = tip_dists / np.maximum(mcp_dists, 1e-6)

    # Thumb-vs-index spread: angle between (wrist→thumb_MCP) and (wrist→index_MCP).
    # Thumbs up tends to have a small angle (thumb pointing up, index curled).
    thumb_dir = norm[THUMB_MCP] - norm[WRIST]
    index_dir = norm[INDEX_MCP] - norm[WRIST]
    thumb_index_angle = _angle_between(thumb_dir[:2], index_dir[:2])

    return np.concatenate([base, tip_dists, extension, [thumb_index_angle]]).astype(np.float32)


def safe_features(
    hand: Optional[HandLandmarks],
) -> Tuple[Optional[np.ndarray], bool]:
    """Return ``(features, hand_present)``.

    If no hand was detected, returns ``(None, False)``. Callers should treat
    no-hand frames as a special class or skip them.
    """
    if hand is None:
        return None, False
    feats = landmarks_to_features(hand)
    return feats, True
