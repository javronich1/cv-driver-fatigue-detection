"""Run MediaPipe over gesture videos and produce a feature dataframe.

For training the per-frame classifier we focus on these folders:

* ``open palm``   → label "open_palm"
* ``thumbs up``   → label "thumbs_up"
* ``random hands``→ label "negative"  (hands visible but not the gesture)
* ``no hands``    → label "negative"  (no hand visible — used to evaluate
  detection robustness, but skipped from feature CSV when no hand is found)

Sequence-level folders (``correct sequence``, ``wrong sequence``,
``too slow``, ``incomplete``) are kept aside for the state-machine evaluation
and not used during per-frame training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import config
from ..data.loader import VideoRecord, discover_gestures
from ..utils.video import iter_frames
from .features import FEATURE_DIM, landmarks_to_features
from .landmarks import HandLandmarkExtractor


# Folder → per-frame training label.
TRAIN_LABEL_BY_FOLDER: Dict[str, str] = {
    "open palm":    "open_palm",
    "thumbs up":    "thumbs_up",
    "random hands": "negative",
    "no hands":     "negative",
}


def _is_training_clip(rec: VideoRecord) -> bool:
    return rec.folder_label in TRAIN_LABEL_BY_FOLDER


def _label_for(rec: VideoRecord) -> str:
    return TRAIN_LABEL_BY_FOLDER[rec.folder_label]


def extract_features_from_video(
    rec: VideoRecord,
    extractor: HandLandmarkExtractor,
    *,
    stride: int = 5,
    resize: Optional[tuple] = (540, 960),
) -> List[Dict]:
    """Return one row per kept frame for one video.

    Frames where MediaPipe finds no hand are still recorded (with
    ``hand_present=False`` and zeroed features) so that the caller can choose
    to either drop them or treat them as a separate class.

    Resizing to 540x960 is a 2x downscale of our 1080x1920 portrait videos —
    much faster, with negligible accuracy loss for hand landmarks.
    """
    rows: List[Dict] = []
    label = _label_for(rec)

    for frame_idx, rgb in iter_frames(
        rec.path, stride=stride, resize=resize, bgr_to_rgb=True
    ):
        hand = extractor.detect(rgb)
        if hand is None:
            row = {
                "video": rec.filename,
                "person": rec.person,
                "folder_label": rec.folder_label,
                "label": label,
                "frame_idx": frame_idx,
                "hand_present": False,
                "hand_score": 0.0,
                "handedness": "",
            }
            row.update({f"f{i:02d}": 0.0 for i in range(FEATURE_DIM)})
            rows.append(row)
            continue

        feats = landmarks_to_features(hand)
        row = {
            "video": rec.filename,
            "person": rec.person,
            "folder_label": rec.folder_label,
            "label": label,
            "frame_idx": frame_idx,
            "hand_present": True,
            "hand_score": hand.score,
            "handedness": hand.handedness,
        }
        row.update({f"f{i:02d}": float(feats[i]) for i in range(FEATURE_DIM)})
        rows.append(row)

    return rows


def build_gesture_feature_table(
    *,
    stride: int = 5,
    resize: Optional[tuple] = (540, 960),
    min_hand_score: float = 0.5,
) -> pd.DataFrame:
    """Run extraction over every training gesture video and return a DataFrame.

    Rows where ``hand_present == False`` for the positive classes are dropped
    (they teach the classifier nothing). For the negative class we keep them
    (they're a valid example of "not a target gesture").
    """
    records = [r for r in discover_gestures() if _is_training_clip(r)]
    rows: List[Dict] = []

    with HandLandmarkExtractor(static_image_mode=True) as extractor:
        for rec in tqdm(records, desc="Extracting hand features"):
            rows.extend(
                extract_features_from_video(
                    rec, extractor, stride=stride, resize=resize
                )
            )

    df = pd.DataFrame(rows)

    # Drop rows where the hand score is too low to be trusted, except for the
    # "negative" class where missing/weak hands are a legitimate signal.
    keep = (df["label"] == "negative") | (
        df["hand_present"] & (df["hand_score"] >= min_hand_score)
    )
    df = df[keep].reset_index(drop=True)
    return df


def save_feature_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path) if path.suffix == ".parquet" else df.to_csv(path, index=False)
