"""Helpers to download and cache pretrained model files (e.g. MediaPipe .task)."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

from .. import config


# Local cache for downloaded MediaPipe .task assets.
MEDIAPIPE_MODELS_DIR = config.PROJECT_ROOT / "models" / "mediapipe"

# Canonical URLs from Google's MediaPipe model zoo.
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n        →   {dest}")
    urllib.request.urlretrieve(url, dest)


def ensure_hand_landmarker(path: Optional[Path] = None) -> Path:
    """Return a local path to the hand_landmarker.task, downloading if needed."""
    target = path or (MEDIAPIPE_MODELS_DIR / "hand_landmarker.task")
    if not target.exists():
        _download(HAND_LANDMARKER_URL, target)
    return target


def ensure_face_landmarker(path: Optional[Path] = None) -> Path:
    target = path or (MEDIAPIPE_MODELS_DIR / "face_landmarker.task")
    if not target.exists():
        _download(FACE_LANDMARKER_URL, target)
    return target
