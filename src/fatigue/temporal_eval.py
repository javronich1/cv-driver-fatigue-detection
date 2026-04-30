"""Per-clip fatigue evaluation with temporal aggregation.

The per-frame model gives noisy frame-level predictions. Fatigue is
fundamentally a temporal phenomenon — a single closed-eye frame is not
drowsiness, but several seconds of low-EAR frames are. We turn frame
probabilities into one decision per clip via two complementary methods:

1. **Mean probability**: average the model's softmax/probability vector
   over every face-frame in the clip, then ``argmax``. Smooth, handles
   class imbalance gracefully.
2. **Rolling-window majority**: take a sliding window of per-frame
   ``argmax`` predictions; each window votes; the clip's decision is the
   majority over all windows. Mirrors how a real-time system would behave.

Evaluation is LOSO, like the per-frame counterpart: for each held-out
person we train on the other person, predict probabilities for the held-
out frames, aggregate per clip, and report per-clip metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.pipeline import Pipeline

from .classical import CLASSES, FEATURE_COLS, get_xy


@dataclass
class ClipPrediction:
    video: str
    person: str
    folder_label: str
    fine_label: str
    coarse_label: str
    n_frames: int
    pred_mean_prob: str
    pred_window_vote: str


@dataclass
class ClipEvalResult:
    fold_name: str
    method: str                     # "mean_prob" or "window_vote"
    accuracy: float
    macro_f1: float
    report: str
    confusion: np.ndarray
    classes: Tuple[str, ...] = CLASSES


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
def aggregate_mean_prob(probs: np.ndarray, classes: Tuple[str, ...]) -> str:
    """Argmax of the mean per-frame probability vector."""
    if probs.size == 0:
        return classes[0]
    avg = probs.mean(axis=0)
    return classes[int(np.argmax(avg))]


def aggregate_window_vote(
    probs: np.ndarray,
    classes: Tuple[str, ...],
    *,
    window: int = 5,
    min_confidence: float = 0.0,
) -> str:
    """Sliding-window majority on per-frame argmax predictions."""
    if probs.size == 0:
        return classes[0]
    pred_idx = probs.argmax(axis=1)
    if min_confidence > 0:
        conf = probs.max(axis=1)
        keep = conf >= min_confidence
        if keep.any():
            pred_idx = pred_idx[keep]
    if len(pred_idx) == 0:
        return classes[0]
    window = max(1, min(window, len(pred_idx)))
    votes = np.zeros(len(classes), dtype=np.int64)
    for i in range(len(pred_idx) - window + 1):
        win = pred_idx[i:i + window]
        c = np.bincount(win, minlength=len(classes)).argmax()
        votes[c] += 1
    return classes[int(np.argmax(votes))]


# ---------------------------------------------------------------------------
# Per-clip LOSO
# ---------------------------------------------------------------------------
def predict_per_clip(
    df_test: pd.DataFrame,
    model: Pipeline,
    *,
    window: int = 5,
) -> pd.DataFrame:
    """Run the trained ``model`` on every face-frame in ``df_test`` and
    aggregate to one prediction per clip with both strategies."""
    classes = tuple(model.classes_)
    X = df_test[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    probs = model.predict_proba(X)

    rows: List[ClipPrediction] = []
    for video, sub in df_test.groupby("video", sort=False):
        idx = sub.index.to_numpy()
        # We need positional indices into X / probs (not df_test's labels).
        pos = df_test.index.get_indexer(idx)
        clip_probs = probs[pos]
        rows.append(ClipPrediction(
            video=video,
            person=str(sub["person"].iloc[0]),
            folder_label=str(sub["folder_label"].iloc[0]),
            fine_label=str(sub["fine_label"].iloc[0]),
            coarse_label=str(sub["coarse_label"].iloc[0]),
            n_frames=len(sub),
            pred_mean_prob=aggregate_mean_prob(clip_probs, classes),
            pred_window_vote=aggregate_window_vote(
                clip_probs, classes, window=window,
            ),
        ))
    return pd.DataFrame([r.__dict__ for r in rows])


def evaluate_clip_level_loso(
    df: pd.DataFrame,
    model_factory: Callable[[], Pipeline],
    *,
    window: int = 5,
) -> Tuple[List[ClipEvalResult], pd.DataFrame]:
    """LOSO clip-level evaluation. Returns ``(results, all_clip_predictions)``.

    Per fold we train on the other person, predict probabilities for every
    held-out face-frame, aggregate per clip with both methods, and score
    accuracy + macro-F1 + a 3x3 confusion against the clip's coarse_label.
    """
    X, y, person = get_xy(df)
    persons = sorted(np.unique(person))
    results: List[ClipEvalResult] = []
    all_clip_preds: List[pd.DataFrame] = []

    for held_out in persons:
        train_mask = person != held_out
        if not train_mask.any():
            continue
        model = model_factory()
        model.fit(X[train_mask], y[train_mask])

        df_test = df[df["person"] == held_out].reset_index(drop=True)
        clip_df = predict_per_clip(df_test, model, window=window)
        clip_df["fold"] = f"test={held_out}"
        all_clip_preds.append(clip_df)

        for method, col in (("mean_prob",   "pred_mean_prob"),
                            ("window_vote", "pred_window_vote")):
            y_true = clip_df["coarse_label"].to_numpy()
            y_pred = clip_df[col].to_numpy()
            cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
            rep = classification_report(
                y_true, y_pred, labels=list(CLASSES), digits=3,
                zero_division=0,
            )
            results.append(ClipEvalResult(
                fold_name=f"test={held_out}",
                method=method,
                accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, labels=list(CLASSES),
                                        average="macro", zero_division=0)),
                report=rep,
                confusion=cm,
            ))

    return results, pd.concat(all_clip_preds, ignore_index=True) \
        if all_clip_preds else pd.DataFrame()
