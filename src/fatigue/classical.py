"""Classical per-frame fatigue classifier (3-class: alert / drowsy / yawning).

Same blueprint as ``src/gestures/classical.py``: an SVM (RBF) and a Random
Forest are trained on the per-frame face features, evaluated with
leave-one-person-out (LOSO), and finally refit on all data so we have a
deployment model.

For a fairer comparison with the gesture classifier we share the same
training conventions:

    * sklearn ``Pipeline`` with ``StandardScaler`` (the SVM is scale-
      sensitive; the RF doesn't care but we keep it for API symmetry).
    * ``class_weight="balanced"`` to compensate for the imbalance
      (drowsy ~58% / alert ~32% / yawning ~10%).
    * Both models expose ``predict_proba`` so the temporal aggregator can
      use confidence smoothing in Stage 4D.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_NAMES


CLASSES: Tuple[str, ...] = ("alert", "drowsy", "yawning")
FEATURE_COLS: Tuple[str, ...] = FEATURE_NAMES


def make_svm(*, class_weight: str = "balanced") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=4.0, gamma="scale",
            class_weight=class_weight,
            probability=True, random_state=42,
        )),
    ])


def make_random_forest(*, class_weight: str = "balanced") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None,
            class_weight=class_weight,
            n_jobs=-1, random_state=42,
        )),
    ])


def get_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, person)``."""
    X = df[list(FEATURE_COLS)].to_numpy(dtype=np.float32)
    y = df["coarse_label"].to_numpy()
    person = df["person"].to_numpy()
    return X, y, person


@dataclass
class FoldResult:
    fold_name: str
    accuracy: float
    macro_f1: float
    report: str
    confusion: np.ndarray
    classes: Tuple[str, ...] = CLASSES


def evaluate_loso(df: pd.DataFrame, model_factory) -> List[FoldResult]:
    X, y, person = get_xy(df)
    persons = sorted(np.unique(person))
    results: List[FoldResult] = []
    for held_out in persons:
        train_mask = person != held_out
        test_mask = person == held_out
        if not test_mask.any() or not train_mask.any():
            continue

        model = model_factory()
        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask])
        y_true = y[test_mask]

        cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
        rep = classification_report(
            y_true, y_pred, labels=list(CLASSES), digits=3, zero_division=0,
        )
        results.append(FoldResult(
            fold_name=f"test={held_out}",
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, labels=list(CLASSES),
                                    average="macro", zero_division=0)),
            report=rep,
            confusion=cm,
        ))
    return results


def fit_on_all(df: pd.DataFrame, model_factory) -> Pipeline:
    X, y, _ = get_xy(df)
    model = model_factory()
    model.fit(X, y)
    return model


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def predict_proba(model: Pipeline, features: np.ndarray) -> Dict[str, float]:
    if features.ndim == 1:
        features = features[None, :]
    probs = model.predict_proba(features)[0]
    classes = list(model.classes_)
    return {c: float(p) for c, p in zip(classes, probs)}
