"""Walk every fatigue clip, run the FaceLandmarker, and produce a per-frame
feature dataframe ready for the classical fatigue classifier.

Each clip's folder name maps to a fine-grained label (see ``GESTURE_FOLDERS``-
style mapping in ``config.py``). For training the classical classifier we
collapse to a coarse 3-class label using ``config.FATIGUE_COARSE_LABEL``:

    alert | drowsy | yawning  (and a held-out ``transition`` class)

Frames where MediaPipe finds no face are still recorded (with
``face_present=False`` and zeroed features) so callers can decide whether
to drop them. The CSV schema:

    video, person, folder_label, fine_label, coarse_label,
    frame_idx, face_present,
    ear_left, ear_right, ear_mean, mar, ...   (24 feature columns)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .. import config
from ..data.loader import VideoRecord, discover_fatigue
from ..utils.video import iter_frames
from .face_landmarks import FaceLandmarkExtractor
from .features import FEATURE_NAMES, landmarks_to_features


def _coarse_label_for(rec: VideoRecord) -> Optional[str]:
    """Look up the coarse label (alert/drowsy/yawning/transition) from
    the canonical fine label, or return None if the folder isn't part of
    the fatigue training set."""
    fine = config.FATIGUE_FOLDERS.get(rec.folder_label)
    if fine is None:
        return None
    return config.FATIGUE_COARSE_LABEL.get(fine)


def extract_features_from_video(
    rec: VideoRecord,
    extractor: FaceLandmarkExtractor,
    *,
    stride: int = 5,
    resize: Optional[Tuple[int, int]] = (540, 960),
) -> List[Dict]:
    """Return one row per kept frame for one video."""
    rows: List[Dict] = []
    fine = config.FATIGUE_FOLDERS.get(rec.folder_label)
    coarse = _coarse_label_for(rec)
    base = {
        "video": rec.filename,
        "person": rec.person,
        "folder_label": rec.folder_label,
        "fine_label": fine,
        "coarse_label": coarse,
    }
    for frame_idx, rgb in iter_frames(
        rec.path, stride=stride, resize=resize, bgr_to_rgb=True
    ):
        face = extractor.detect(rgb)
        row = dict(base)
        row["frame_idx"] = frame_idx
        if face is None:
            row["face_present"] = False
            row.update({n: 0.0 for n in FEATURE_NAMES})
        else:
            row["face_present"] = True
            feats = landmarks_to_features(face)
            row.update({n: float(feats[i]) for i, n in enumerate(FEATURE_NAMES)})
        rows.append(row)
    return rows


def build_fatigue_feature_table(
    *,
    stride: int = 5,
    resize: Optional[Tuple[int, int]] = (540, 960),
    drop_missing_face: bool = True,
    include_transition: bool = False,
) -> pd.DataFrame:
    """Run extraction over every fatigue clip and return a DataFrame.

    Frames where ``face_present == False`` are dropped by default — the
    classical model is trained on faces only. Set ``include_transition``
    to True to keep the ``transition`` coarse class (held out from training
    by default but useful for temporal evaluation).
    """
    records: List[VideoRecord] = []
    for r in discover_fatigue():
        c = _coarse_label_for(r)
        if c is None:
            continue
        if c == "transition" and not include_transition:
            continue
        records.append(r)

    rows: List[Dict] = []
    with FaceLandmarkExtractor(static_image_mode=True) as extractor:
        for rec in tqdm(records, desc="Extracting face features"):
            rows.extend(extract_features_from_video(
                rec, extractor, stride=stride, resize=resize,
            ))

    df = pd.DataFrame(rows)
    if drop_missing_face and "face_present" in df.columns:
        df = df[df["face_present"]].reset_index(drop=True)
    return df


def save_feature_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)
