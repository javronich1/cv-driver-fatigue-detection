"""MediaPipe FaceLandmarker wrapper, analogous to the hand version.

Returns dense 478-point face landmarks plus blendshapes (52 predefined
expression scores including ``eyeBlinkLeft``, ``eyeBlinkRight``, ``jawOpen``,
``mouthSmile``, ...) and a 4x4 head transformation matrix that we use for
pitch/yaw/roll head-pose estimation.

Usage::

    with FaceLandmarkExtractor() as ex:
        face = ex.detect(rgb_frame)
        if face is not None:
            print(face.coords.shape)        # (478, 3)
            print(face.blendshape("jawOpen"))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import mediapipe as mp  # type: ignore
from mediapipe.tasks import python as mp_python  # type: ignore
from mediapipe.tasks.python import vision as mp_vision  # type: ignore

from ..utils.models import ensure_face_landmarker


N_LANDMARKS = 478


@dataclass(frozen=True)
class FaceLandmarks:
    """Container for one detected face's landmarks + blendshapes + head pose.

    Attributes
    ----------
    coords : np.ndarray, shape (478, 3)
        Normalized x, y, z coordinates from MediaPipe.
    blendshapes : dict[str, float]
        Blendshape category name → score in [0, 1].
    transform : np.ndarray, shape (4, 4)
        Facial transformation matrix (rotation + translation in camera space).
        Identity if MediaPipe didn't return one.
    score : float
        Detection confidence in [0, 1] (best available proxy).
    """
    coords: np.ndarray
    blendshapes: Dict[str, float] = field(default_factory=dict)
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    score: float = 1.0

    def blendshape(self, name: str, default: float = 0.0) -> float:
        return float(self.blendshapes.get(name, default))


class FaceLandmarkExtractor:
    """Wrapper around mediapipe.tasks.vision.FaceLandmarker."""

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        static_image_mode: bool = True,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
    ) -> None:
        model_path = ensure_face_landmarker(model_path)
        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
        running_mode = (
            mp_vision.RunningMode.IMAGE if static_image_mode
            else mp_vision.RunningMode.VIDEO
        )
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._static = static_image_mode
        self._video_ts_ms = 0

    def detect(
        self,
        rgb_frame: np.ndarray,
        *,
        timestamp_ms: Optional[int] = None,
    ) -> Optional[FaceLandmarks]:
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            return None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._static:
            result = self._landmarker.detect(mp_image)
        else:
            if timestamp_ms is None:
                self._video_ts_ms += 33
                timestamp_ms = self._video_ts_ms
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        # Single face only.
        lm = result.face_landmarks[0]
        coords = np.array(
            [[p.x, p.y, p.z] for p in lm], dtype=np.float32,
        )

        bs: Dict[str, float] = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                bs[cat.category_name] = float(cat.score)

        transform = np.eye(4, dtype=np.float32)
        if result.facial_transformation_matrixes:
            transform = np.array(
                result.facial_transformation_matrixes[0], dtype=np.float32,
            )

        # FaceLandmarker doesn't expose a single per-detection score, so we
        # use the average blendshape activation of "neutral-ish" mouthClose
        # (cheap proxy in [0, 1]); else 1.0.
        score = float(bs.get("_neutral", 1.0))

        return FaceLandmarks(
            coords=coords,
            blendshapes=bs,
            transform=transform,
            score=score,
        )

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None  # type: ignore

    def __enter__(self) -> "FaceLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
