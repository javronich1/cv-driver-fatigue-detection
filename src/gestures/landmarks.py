"""Hand landmark extraction with MediaPipe Tasks API (mp.tasks.vision).

The legacy ``mp.solutions.hands`` API was dropped in MediaPipe 0.10.21+ on
Python 3.12, so we use the modern ``HandLandmarker`` from
``mp.tasks.vision``. It needs a ``.task`` model file which is auto-downloaded
on first use into ``models/mediapipe/``.

Usage::

    with HandLandmarkExtractor() as ex:
        result = ex.detect(rgb_frame)        # returns Optional[HandLandmarks]
        if result is not None:
            print(result.coords.shape)       # (21, 3)

We expose the same dataclass / API as before so the rest of the project
doesn't need to know about MediaPipe versions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

import mediapipe as mp  # type: ignore
from mediapipe.tasks import python as mp_python  # type: ignore
from mediapipe.tasks.python import vision as mp_vision  # type: ignore

from ..utils.models import ensure_hand_landmarker


# MediaPipe constants
N_LANDMARKS = 21
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


@dataclass(frozen=True)
class HandLandmarks:
    """Container for one detected hand's landmarks.

    Attributes
    ----------
    coords : np.ndarray, shape (21, 3)
        Normalized x, y, z coordinates from MediaPipe.
    handedness : str
        "Left" or "Right" as predicted by MediaPipe.
    score : float
        Detection confidence in [0, 1].
    """
    coords: np.ndarray
    handedness: str
    score: float


class HandLandmarkExtractor:
    """Wrapper around mediapipe.tasks.vision.HandLandmarker.

    Parameters
    ----------
    model_path : Path, optional
        Override the default model location.
    static_image_mode : bool
        If True, every call is treated as an unrelated still image (more
        accurate, slightly slower). If False, runs in VIDEO mode with tracking.
    min_detection_confidence : float
        Filter detections below this score before returning.
    max_num_hands : int
        Number of hands to detect (we use 1).
    """

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        static_image_mode: bool = True,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
    ) -> None:
        model_path = ensure_hand_landmarker(model_path)
        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
        running_mode = (
            mp_vision.RunningMode.IMAGE if static_image_mode
            else mp_vision.RunningMode.VIDEO
        )
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._static = static_image_mode
        self._video_ts_ms = 0

    def detect(
        self,
        rgb_frame: np.ndarray,
        *,
        timestamp_ms: Optional[int] = None,
    ) -> Optional[HandLandmarks]:
        """Run hand detection on an RGB frame.

        For VIDEO mode, ``timestamp_ms`` must be strictly increasing across
        calls. If omitted we auto-increment by ~33 ms (≈30 fps).
        """
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            return None

        # MediaPipe wants an mp.Image with SRGB format.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._static:
            result = self._landmarker.detect(mp_image)
        else:
            if timestamp_ms is None:
                self._video_ts_ms += 33
                timestamp_ms = self._video_ts_ms
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        # Pick highest-confidence hand.
        best_idx = 0
        best_score = -1.0
        for i, h in enumerate(result.handedness or []):
            score = h[0].score
            if score > best_score:
                best_score = score
                best_idx = i

        lm = result.hand_landmarks[best_idx]
        coords = np.array(
            [[p.x, p.y, p.z] for p in lm], dtype=np.float32
        )
        handed = (
            result.handedness[best_idx][0].category_name
            if result.handedness else "Right"
        )
        return HandLandmarks(coords=coords, handedness=handed, score=float(best_score))

    def detect_many(
        self, rgb_frames: List[np.ndarray]
    ) -> List[Optional[HandLandmarks]]:
        return [self.detect(f) for f in rgb_frames]

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None  # type: ignore

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
