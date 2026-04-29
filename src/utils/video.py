"""Video utilities: probing metadata, iterating frames, sampling.

OpenCV is used as the backend. All functions are tolerant to missing/corrupt
videos: they return ``None`` (for ``probe``) or yield nothing (for iterators)
rather than raising, so a single bad file does not break a long pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2  # type: ignore
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Lightweight metadata for a video file."""
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    fourcc: str
    readable: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def _fourcc_to_str(code: int) -> str:
    try:
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return ""


def probe(video_path: Path) -> Optional[VideoInfo]:
    """Return basic metadata for a video, or ``None`` if it can't be opened."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))

        # Quick read sanity check: try to grab the first frame.
        ok, _ = cap.read()
        readable = bool(ok)

        duration_s = (n_frames / fps) if fps > 0 else 0.0
        return VideoInfo(
            path=video_path,
            width=width, height=height, fps=fps,
            n_frames=n_frames, duration_s=duration_s,
            fourcc=fourcc, readable=readable,
        )
    finally:
        cap.release()


def iter_frames(
    video_path: Path,
    *,
    stride: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    bgr_to_rgb: bool = True,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame)`` pairs from a video.

    Parameters
    ----------
    stride : int
        Keep every Nth frame (1 = every frame).
    max_frames : int, optional
        Stop after this many *kept* frames.
    resize : (w, h), optional
        Resize each yielded frame to this size.
    bgr_to_rgb : bool
        OpenCV decodes BGR; convert to RGB by default (most ML libs expect RGB).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    kept = 0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                if bgr_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield idx, frame
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def sample_frame(
    video_path: Path,
    fraction: float = 0.5,
    *,
    bgr_to_rgb: bool = True,
) -> Optional[np.ndarray]:
    """Return a single frame at the given fractional position (0..1)."""
    info = probe(video_path)
    if info is None or info.n_frames <= 0:
        return None
    target = max(0, min(info.n_frames - 1, int(info.n_frames * fraction)))
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            return None
        if bgr_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    finally:
        cap.release()
