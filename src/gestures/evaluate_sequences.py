"""End-to-end evaluation: run the full classical gesture pipeline (MediaPipe
landmarks + classifier + state machine) on the sequence-level test folders.

Ground-truth mapping:

    correct sequence  → should_activate = True
    wrong sequence    → should_activate = False
    too slow          → should_activate = False
    incomplete        → should_activate = False
    random hands      → should_activate = False  (sanity check)
    no hands          → should_activate = False  (sanity check)

For each clip we feed every probed frame through the pipeline and record
whether the state machine reached ``ACTIVATED`` before the clip ended.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .. import config
from ..data.loader import VideoRecord, discover_gestures
from ..utils.video import iter_frames, probe
from .features import FEATURE_DIM, landmarks_to_features
from .landmarks import HandLandmarkExtractor
from .state_machine import (
    FrameEvent, GestureSequenceStateMachine, StateMachineConfig,
)


SEQUENCE_GT = {
    "correct sequence": True,
    "wrong sequence":   False,
    "too slow":         False,
    "incomplete":       False,
    "random hands":     False,
    "no hands":         False,
}


@dataclass
class ClipResult:
    folder_label: str
    person: Optional[str]
    filename: str
    duration_s: float
    n_frames_probed: int
    n_hand_frames: int
    activated: bool
    should_activate: bool
    correct: bool


def _classify_frame(
    feats: np.ndarray,
    model: Pipeline,
) -> tuple[str, float]:
    probs = model.predict_proba(feats[None, :])[0]
    classes = list(model.classes_)
    j = int(np.argmax(probs))
    return classes[j], float(probs[j])


def evaluate_clip(
    rec: VideoRecord,
    extractor: HandLandmarkExtractor,
    model: Pipeline,
    sm_config: StateMachineConfig,
    *,
    stride: int = 5,
    resize: tuple = (540, 960),
) -> ClipResult:
    info = probe(rec.path)
    duration = info.duration_s if info else 0.0
    fps = info.fps if info and info.fps else 30.0

    sm = GestureSequenceStateMachine(sm_config)
    n_probed = 0
    n_hand = 0
    activated = False

    for frame_idx, rgb in iter_frames(rec.path, stride=stride, resize=resize, bgr_to_rgb=True):
        n_probed += 1
        timestamp_s = frame_idx / fps if fps > 0 else 0.0

        hand = extractor.detect(rgb)
        if hand is None:
            ev = FrameEvent(timestamp_s=timestamp_s, label="negative", confidence=1.0)
        else:
            n_hand += 1
            feats = landmarks_to_features(hand)
            label, conf = _classify_frame(feats, model)
            ev = FrameEvent(timestamp_s=timestamp_s, label=label, confidence=conf)

        decision = sm.update(ev)
        if decision.activated:
            activated = True
            break  # state is sticky once activated

    should = SEQUENCE_GT.get(rec.folder_label, False)
    return ClipResult(
        folder_label=rec.folder_label,
        person=rec.person,
        filename=rec.filename,
        duration_s=round(duration, 3),
        n_frames_probed=n_probed,
        n_hand_frames=n_hand,
        activated=activated,
        should_activate=should,
        correct=(activated == should),
    )


def evaluate_all(
    model: Pipeline,
    sm_config: Optional[StateMachineConfig] = None,
    *,
    stride: int = 5,
    resize: tuple = (540, 960),
    progress: bool = True,
) -> pd.DataFrame:
    sm_config = sm_config or StateMachineConfig()
    records = [
        r for r in discover_gestures()
        if r.folder_label in SEQUENCE_GT
    ]
    rows: List[ClipResult] = []
    iterable = tqdm(records, desc="Sequence eval") if progress else records
    with HandLandmarkExtractor(static_image_mode=True) as extractor:
        for rec in iterable:
            rows.append(
                evaluate_clip(
                    rec, extractor, model, sm_config,
                    stride=stride, resize=resize,
                )
            )
    return pd.DataFrame([asdict(r) for r in rows])


def summarise(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("GESTURE SEQUENCE — END-TO-END EVALUATION")
    lines.append("=" * 72)

    total = len(df)
    correct = int(df["correct"].sum())
    overall_acc = correct / total if total else 0.0
    lines.append(f"Total clips    : {total}")
    lines.append(f"Correct        : {correct}")
    lines.append(f"Overall acc    : {overall_acc:.3f}")
    lines.append("")

    lines.append(f"{'folder':<22}{'n':>4}{'activated':>11}"
                 f"{'should_act':>12}{'acc':>8}")
    for folder, sub in df.groupby("folder_label"):
        n = len(sub)
        n_act = int(sub["activated"].sum())
        gt = bool(sub["should_activate"].iloc[0])
        acc = float(sub["correct"].mean())
        lines.append(
            f"{folder:<22}{n:>4}{n_act:>11}{str(gt):>12}{acc:>8.3f}"
        )
    lines.append("")

    # Confusion-style: TP / FP / FN / TN where positive == "should_activate"
    pos = df["should_activate"]
    pred = df["activated"]
    tp = int(((pos) & (pred)).sum())
    fp = int(((~pos) & (pred)).sum())
    fn = int(((pos) & (~pred)).sum())
    tn = int(((~pos) & (~pred)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    lines.append("Activation as binary classification (positive = should_activate):")
    lines.append(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    lines.append(f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    lines.append("=" * 72)
    return "\n".join(lines)
