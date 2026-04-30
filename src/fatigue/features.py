"""Per-frame feature engineering for the classical fatigue pipeline.

We turn one ``FaceLandmarks`` detection into a fixed-length feature vector
combining classical geometric cues with MediaPipe blendshapes:

    Geometric (computed from the 478 landmarks)
        ear_left, ear_right, ear_mean         (eye aspect ratio)
        mar                                    (mouth aspect ratio)
        eye_open_left, eye_open_right          (vertical eye gap / face size)
        mouth_open                             (vertical mouth gap / face size)

    Head pose (Euler angles in degrees, decomposed from the 4x4
    transformation matrix returned by FaceLandmarker)
        head_pitch, head_yaw, head_roll

    Blendshapes (MediaPipe FaceLandmarker, scores in [0, 1])
        eyeBlinkLeft, eyeBlinkRight
        eyeSquintLeft, eyeSquintRight
        eyeLookDownLeft, eyeLookDownRight
        jawOpen, mouthOpen, mouthFunnel, mouthPucker
        browDownLeft, browDownRight
        cheekSquintLeft, cheekSquintRight

The list is kept short and well-known so the classical model stays
interpretable. ``FEATURE_NAMES`` is the canonical column order.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .face_landmarks import FaceLandmarks


# MediaPipe Face Mesh / Face Landmarker uses the same 468/478-point topology.
# These are the canonical 6-point EAR landmark indices (Soukupová & Cech 2016
# adapted for MediaPipe Face Mesh).
EAR_RIGHT_IDX = (33, 160, 158, 133, 153, 144)
EAR_LEFT_IDX = (362, 385, 387, 263, 373, 380)

# Mouth — outer corners + top/bottom of the lip line.
MOUTH_RIGHT_CORNER = 78
MOUTH_LEFT_CORNER = 308
MOUTH_TOP = 13           # upper lip centre
MOUTH_BOTTOM = 14        # lower lip centre

# Used for face-size normalisation (interpupillary-ish distance).
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


# Order matters: FEATURE_NAMES is the column schema of the CSV.
GEOMETRIC_NAMES: Tuple[str, ...] = (
    "ear_left", "ear_right", "ear_mean",
    "mar",
    "eye_open_left", "eye_open_right",
    "mouth_open",
)

HEAD_POSE_NAMES: Tuple[str, ...] = (
    "head_pitch", "head_yaw", "head_roll",
)

BLENDSHAPE_NAMES: Tuple[str, ...] = (
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
    "browDownLeft", "browDownRight",
    "cheekSquintLeft", "cheekSquintRight",
)

FEATURE_NAMES: Tuple[str, ...] = (
    GEOMETRIC_NAMES + HEAD_POSE_NAMES + BLENDSHAPE_NAMES
)
FEATURE_DIM = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(coords: np.ndarray, idx: Tuple[int, ...]) -> float:
    """6-point EAR: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""
    p1, p2, p3, p4, p5, p6 = (coords[i] for i in idx)
    horiz = _euclid(p1, p4)
    if horiz < 1e-6:
        return 0.0
    vert = _euclid(p2, p6) + _euclid(p3, p5)
    return vert / (2.0 * horiz)


def _mouth_aspect_ratio(coords: np.ndarray) -> float:
    """Vertical / horizontal mouth opening."""
    horiz = _euclid(coords[MOUTH_RIGHT_CORNER], coords[MOUTH_LEFT_CORNER])
    if horiz < 1e-6:
        return 0.0
    vert = _euclid(coords[MOUTH_TOP], coords[MOUTH_BOTTOM])
    return vert / horiz


def _face_size(coords: np.ndarray) -> float:
    """Outer-eye-to-outer-eye distance — a stable face-scale proxy."""
    return _euclid(coords[LEFT_EYE_OUTER], coords[RIGHT_EYE_OUTER])


def _euler_from_matrix(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a 3x3 rotation matrix into XYZ Euler angles (deg).

    We follow the MediaPipe convention where the matrix is camera → face,
    so:
        pitch = rotation about X (nodding 'yes')
        yaw   = rotation about Y (shaking 'no')
        roll  = rotation about Z (tilting head sideways)
    """
    R = matrix[:3, :3]
    # Reference: standard XYZ extraction.
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        x = float(np.arctan2(R[2, 1], R[2, 2]))
        y = float(np.arctan2(-R[2, 0], sy))
        z = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        x = float(np.arctan2(-R[1, 2], R[1, 1]))
        y = float(np.arctan2(-R[2, 0], sy))
        z = 0.0
    return (np.degrees(x), np.degrees(y), np.degrees(z))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def landmarks_to_features(face: FaceLandmarks) -> np.ndarray:
    """Return a 1-D float32 vector of size :data:`FEATURE_DIM`."""
    coords = face.coords
    out = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 1) Geometric.
    ear_l = _eye_aspect_ratio(coords, EAR_LEFT_IDX)
    ear_r = _eye_aspect_ratio(coords, EAR_RIGHT_IDX)
    mar = _mouth_aspect_ratio(coords)
    fs = _face_size(coords)
    fs_safe = max(fs, 1e-6)
    eye_open_l = _euclid(coords[EAR_LEFT_IDX[1]], coords[EAR_LEFT_IDX[5]]) / fs_safe
    eye_open_r = _euclid(coords[EAR_RIGHT_IDX[1]], coords[EAR_RIGHT_IDX[5]]) / fs_safe
    mouth_open = _euclid(coords[MOUTH_TOP], coords[MOUTH_BOTTOM]) / fs_safe

    geom = (
        ear_l, ear_r, (ear_l + ear_r) / 2.0,
        mar,
        eye_open_l, eye_open_r,
        mouth_open,
    )

    # 2) Head pose (degrees).
    pitch, yaw, roll = _euler_from_matrix(face.transform)

    # 3) Blendshapes (default 0 if missing).
    bs = tuple(face.blendshape(name) for name in BLENDSHAPE_NAMES)

    values = geom + (pitch, yaw, roll) + bs
    assert len(values) == FEATURE_DIM, f"feature mismatch: {len(values)} vs {FEATURE_DIM}"
    out[:] = values
    return out


def safe_features(face: object) -> Tuple[np.ndarray, bool]:
    """Return ``(features, face_present)``. Zeroed features when no face."""
    if face is None or not isinstance(face, FaceLandmarks):
        return np.zeros(FEATURE_DIM, dtype=np.float32), False
    return landmarks_to_features(face), True
