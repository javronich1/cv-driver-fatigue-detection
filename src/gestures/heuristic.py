"""Data-free geometric gesture classifier (Stage G-H).

Why this exists
---------------
The trained SVM/RF gesture classifiers achieve 80 % LOSO macro-F1 in
the report, but ``open_palm`` recall on held-out subjects is only ~50 %
and at *deployment* time on a new subject + new lighting the SVM
collapses to outputting `negative` ≈0.99 even on a clearly held-up
palm. With only 2 training subjects there is simply not enough data
to learn a robust palm/thumb classifier — the same reason our fatigue
heuristic outperforms the trained models.

This module implements a **rule-based** classifier on top of the
MediaPipe 21-point hand landmark layout. It is invariant to:

* translation (we re-center on the wrist)
* scale (we measure everything relative to the wrist→middle-MCP
  distance — a robust hand-size proxy)
* handedness (left hands are mirrored before measuring)

It is intended for the realtime activation step. The trained
SVM/RF/CNN classifiers are kept for the LOSO comparison in the
report, but the demo defaults to this heuristic because it actually
works on unseen subjects.

Rules
-----
A finger is "extended" if its tip is far from the wrist relative to
its MCP joint, AND the tip is "above" (in image space, lower y) the
PIP joint. The thumb has its own rule because it bends sideways
rather than down.

* **open_palm**:  index, middle, ring, pinky all extended.
                  Either thumb extended or wrist roughly below the
                  fingertips (palm-up orientation).
* **thumbs_up**:  thumb extended AND pointing up (tip y < MCP y by
                  a clear margin); index/middle/ring/pinky NOT
                  extended (curled).
* **negative**:   anything else.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .landmarks import (
    FINGER_MCPS, FINGER_TIPS, HandLandmarks,
    INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP,
    PINKY_MCP, PINKY_TIP, RING_MCP, RING_TIP,
    THUMB_MCP, THUMB_IP, THUMB_TIP, WRIST,
    INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP,
)


GESTURE_CLASSES = ("open_palm", "thumbs_up", "negative")


def _normalize(coords: np.ndarray, mirror: bool) -> np.ndarray:
    """Wrist-centered, hand-size-normalized 3D coords."""
    if mirror:
        coords = coords.copy()
        coords[:, 0] = 1.0 - coords[:, 0]
    centered = coords - coords[WRIST]
    hand_size = np.linalg.norm(centered[MIDDLE_MCP])
    if hand_size < 1e-6:
        return centered
    return centered / hand_size


def _extension_ratios(norm: np.ndarray) -> np.ndarray:
    """``tip_dist / mcp_dist`` per finger (>~1.4 ⇒ extended)."""
    tips = np.linalg.norm(norm[list(FINGER_TIPS)], axis=1)
    mcps = np.linalg.norm(norm[list(FINGER_MCPS)], axis=1)
    return tips / np.maximum(mcps, 1e-6)


def _classify(hand: HandLandmarks) -> Tuple[str, Dict[str, float]]:
    mirror = hand.handedness == "Left"
    norm = _normalize(hand.coords, mirror=mirror)
    ext = _extension_ratios(norm)              # thumb, index, mid, ring, pinky
    thumb_ext, index_ext, mid_ext, ring_ext, pinky_ext = ext

    # In image space, smaller y == higher on screen.
    # "Pointing up" tests: tip y is meaningfully less than the PIP/MCP joint.
    def _points_up(tip: int, base: int, *, margin: float = 0.10) -> bool:
        return (norm[base, 1] - norm[tip, 1]) > margin

    fingers_extended = (
        index_ext > 1.55,
        mid_ext > 1.55,
        ring_ext > 1.55,
        pinky_ext > 1.45,
    )
    n_fingers_extended = sum(fingers_extended)

    fingers_up = (
        _points_up(INDEX_TIP, INDEX_PIP),
        _points_up(MIDDLE_TIP, MIDDLE_PIP),
        _points_up(RING_TIP, RING_PIP),
        _points_up(PINKY_TIP, PINKY_PIP),
    )
    n_fingers_up = sum(fingers_up)

    fingers_curled = (
        index_ext < 1.30,
        mid_ext < 1.30,
        ring_ext < 1.30,
        pinky_ext < 1.30,
    )
    n_fingers_curled = sum(fingers_curled)

    thumb_pointing_up = _points_up(THUMB_TIP, THUMB_MCP, margin=0.30)
    thumb_extended = thumb_ext > 1.40

    # ---------------- Scoring ----------------
    # The two target gestures are mutually exclusive in the limit:
    # open_palm has *all* fingers extended; thumbs_up has *all but the
    # thumb* curled. We score each and rely on the curled-vs-extended
    # contrast to keep them apart.
    palm_score = (
        0.45 * (n_fingers_extended / 4.0)
        + 0.35 * (n_fingers_up / 4.0)
        + 0.10 * float(thumb_extended)
        + 0.05
    )

    # Thumbs up REQUIRES the four fingers to be curled. Without that
    # condition, "open palm with thumb up" would wrongly trigger the
    # thumbs-up branch.
    thumb_score = (
        0.40 * float(thumb_pointing_up and n_fingers_curled >= 3)
        + 0.40 * (n_fingers_curled / 4.0)
        + 0.10 * float(thumb_extended)
        + 0.05
    )

    # Hard rules to push confidence up when the gesture is unambiguous.
    if n_fingers_extended >= 3 and n_fingers_up >= 3:
        palm_score = max(palm_score, 0.90)
        # An open palm cannot also be a thumbs up — kill thumb score.
        thumb_score = min(thumb_score, 0.20)
    if thumb_pointing_up and n_fingers_curled >= 3:
        thumb_score = max(thumb_score, 0.90)
        palm_score = min(palm_score, 0.20)

    # Negative = how unlike either gesture this hand is.
    neg_score = max(0.05, 1.0 - max(palm_score, thumb_score))

    raw = {
        "open_palm": float(palm_score),
        "thumbs_up": float(thumb_score),
        "negative": float(neg_score),
    }
    total = sum(raw.values())
    probs = {k: v / total for k, v in raw.items()}
    label = max(probs, key=probs.get)
    return label, probs


def predict(hand: HandLandmarks) -> Tuple[str, float, Dict[str, float]]:
    """Return ``(label, confidence, full_probs)`` for one hand."""
    label, probs = _classify(hand)
    return label, float(probs[label]), probs
