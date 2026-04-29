"""Classical per-frame gesture classifier.

We train two classifiers (SVM and Random Forest) on the 74-dim landmark
features and compare them. Both are wrapped in a Pipeline with
``StandardScaler`` (SVM is scale-sensitive; RF is not but it doesn't hurt).

Evaluation strategy: **leave-one-person-out** cross-validation. With only
two subjects, this means two folds:

    fold 1: train on person1, test on person2
    fold 2: train on person2, test on person1

We report per-class precision/recall/F1, the macro-F1, and confusion
matrices. We also fit a final model on **both** persons combined to deploy
in the real-time system.
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
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_DIM


CLASSES = ("open_palm", "thumbs_up", "negative")
FEATURE_COLS = [f"f{i:02d}" for i in range(FEATURE_DIM)]


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
        ("scaler", StandardScaler()),  # harmless for RF, kept for consistency
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight=class_weight,
            n_jobs=-1, random_state=42,
        )),
    ])


def get_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, person)`` arrays from the feature dataframe."""
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    person = df["person"].to_numpy()
    return X, y, person


@dataclass
class FoldResult:
    fold_name: str
    accuracy: float
    macro_f1: float
    report: str
    confusion: np.ndarray
    classes: Tuple[str, ...]


def evaluate_loso(
    df: pd.DataFrame,
    model_factory,
) -> List[FoldResult]:
    """Leave-one-person-out evaluation across ``df``."""
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
            classes=CLASSES,
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
    """Return ``{label: prob}`` for a single feature vector."""
    if features.ndim == 1:
        features = features[None, :]
    probs = model.predict_proba(features)[0]
    classes = list(model.classes_)
    return {c: float(p) for c, p in zip(classes, probs)}
