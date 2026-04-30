"""Hand-crop extraction for the modern (CNN) gesture pipeline.

For each training gesture clip we sample frames at the configured stride,
run MediaPipe to obtain a hand bounding box, expand it with a small margin,
crop and resize to a square thumbnail, and write the result to disk along
with a CSV index that mirrors the layout used by the classical pipeline.

Crops are written into::

    data_processed/hand_crops/{label}/{person}_{video_stem}_{frame_idx:06d}.png

The CSV index (``data_processed/hand_crops/index.csv``) records:

    label, person, folder_label, video, frame_idx, hand_score, handedness,
    bbox_x0, bbox_y0, bbox_x1, bbox_y1, crop_path
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import config
from ..data.loader import VideoRecord, discover_gestures
from ..utils.video import iter_frames
from .extraction import TRAIN_LABEL_BY_FOLDER
from .landmarks import HandLandmarkExtractor, HandLandmarks


CROP_DIR = config.PROCESSED_DIR / "hand_crops"
CROP_INDEX = CROP_DIR / "index.csv"
DEFAULT_CROP_SIZE = 96  # square hand thumbnail
DEFAULT_MARGIN = 0.30   # 30% padding around the landmark bbox


@dataclass
class CropRecord:
    label: str
    person: Optional[str]
    folder_label: str
    video: str
    frame_idx: int
    hand_score: float
    handedness: str
    bbox_x0: int
    bbox_y0: int
    bbox_x1: int
    bbox_y1: int
    crop_path: str


# ---------------------------------------------------------------------------
# Bounding box from landmarks
# ---------------------------------------------------------------------------
def landmark_bbox(
    hand: HandLandmarks,
    image_size: Tuple[int, int],
    *,
    margin: float = DEFAULT_MARGIN,
) -> Tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` pixel bbox enclosing the hand landmarks.

    ``image_size`` is ``(height, width)``. Coordinates are normalized so we
    convert to pixels, take the min/max, square the box (so the crop has the
    same aspect ratio as the model input), and apply a percentage margin.
    """
    h, w = image_size[:2]
    xs = hand.coords[:, 0] * w
    ys = hand.coords[:, 1] * h

    x0 = float(xs.min())
    x1 = float(xs.max())
    y0 = float(ys.min())
    y1 = float(ys.max())

    # Square the box around its centre so the resize doesn't distort the hand.
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0)
    side *= (1.0 + 2.0 * margin)

    half = side / 2.0
    x0 = int(round(cx - half))
    x1 = int(round(cx + half))
    y0 = int(round(cy - half))
    y1 = int(round(cy + half))

    # Clamp to image bounds — if clamped we lose some margin, that's fine.
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0
    return x0, y0, x1, y1


def crop_hand(
    rgb_frame: np.ndarray,
    hand: HandLandmarks,
    *,
    out_size: int = DEFAULT_CROP_SIZE,
    margin: float = DEFAULT_MARGIN,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Return ``(crop_rgb, bbox_xyxy)`` or ``None`` if the bbox is degenerate."""
    bbox = landmark_bbox(hand, rgb_frame.shape[:2], margin=margin)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    crop = rgb_frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop, bbox


# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------
# CNN training set uses the *positive* folders + "random hands" only.
# "no hands" is excluded — by definition there is no hand to crop, and at
# inference the CNN only runs when MediaPipe found a hand (else the state
# machine receives a forced "negative" event).
CNN_FOLDER_LABELS: Dict[str, str] = {
    "open palm":    "open_palm",
    "thumbs up":    "thumbs_up",
    "random hands": "negative",
}


def _is_cnn_clip(rec: VideoRecord) -> bool:
    return rec.folder_label in CNN_FOLDER_LABELS


def build_hand_crop_dataset(
    *,
    stride: int = 5,
    resize: Tuple[int, int] = (540, 960),
    out_size: int = DEFAULT_CROP_SIZE,
    margin: float = DEFAULT_MARGIN,
    min_hand_score: float = 0.5,
    out_dir: Path = CROP_DIR,
    save_images: bool = True,
) -> pd.DataFrame:
    """Walk every training gesture clip and write a hand-crop dataset.

    Returns a dataframe of the index (also saved to ``index.csv``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for lbl in CNN_FOLDER_LABELS.values():
        (out_dir / lbl).mkdir(parents=True, exist_ok=True)

    records = [r for r in discover_gestures() if _is_cnn_clip(r)]
    rows: List[CropRecord] = []

    with HandLandmarkExtractor(static_image_mode=True) as extractor:
        for rec in tqdm(records, desc="Cropping hands"):
            label = CNN_FOLDER_LABELS[rec.folder_label]
            stem = Path(rec.filename).stem.replace(" ", "_")
            for frame_idx, rgb in iter_frames(
                rec.path, stride=stride, resize=resize, bgr_to_rgb=True
            ):
                hand = extractor.detect(rgb)
                if hand is None or hand.score < min_hand_score:
                    continue
                cropped = crop_hand(rgb, hand, out_size=out_size, margin=margin)
                if cropped is None:
                    continue
                crop_rgb, bbox = cropped

                rel_name = f"{rec.person or 'unknown'}_{stem}_{frame_idx:06d}.png"
                crop_path = out_dir / label / rel_name
                if save_images:
                    # cv2 expects BGR.
                    cv2.imwrite(
                        str(crop_path),
                        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                    )

                rows.append(CropRecord(
                    label=label,
                    person=rec.person,
                    folder_label=rec.folder_label,
                    video=rec.filename,
                    frame_idx=frame_idx,
                    hand_score=float(hand.score),
                    handedness=hand.handedness,
                    bbox_x0=bbox[0], bbox_y0=bbox[1],
                    bbox_x1=bbox[2], bbox_y1=bbox[3],
                    crop_path=str(crop_path.relative_to(config.PROJECT_ROOT)),
                ))

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


def save_index(df: pd.DataFrame, path: Path = CROP_INDEX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
