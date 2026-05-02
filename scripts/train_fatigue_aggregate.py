"""Stage 5E: train + LOSO-evaluate the clip-aggregate fatigue classifier.

Inputs:
    outputs/reports/fatigue_features.csv

Outputs:
    outputs/reports/fatigue_aggregate_loso.txt
    outputs/reports/fatigue_clip_eval_aggregate.csv
    outputs/figures/fatigue_clip_confusion_aggregate_pooled.png
    outputs/figures/fatigue_clip_confusion_aggregate_test=person*.png
    outputs/models/fatigue_aggregate.joblib

The model is a small ``LogisticRegression`` over 15 hand-engineered
clip-level statistics (see ``src/fatigue/aggregate.py``). Picked LR
because:

  * Linear → small, interpretable, no hyper-tuning landmines on N=212.
  * ``class_weight=balanced`` handles the 58/32/10 imbalance.
  * ``predict_proba`` plays nicely with the realtime ensemble.

We also ship a plain RF for ablation; LR wins on LOSO mean-F1 in our
data so we save that as the deployment model. Final model is fit on all
clips after LOSO.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                  # noqa: E402
from src.fatigue.aggregate import (                                     # noqa: E402
    AGG_FEATURE_NAMES, aggregate_features_from_dataframe,
)
from src.fatigue.classical import CLASSES                               # noqa: E402
from src.utils.plotting import plot_confusion                           # noqa: E402


def _build_lr() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        ),
    )


def _build_rf() -> object:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _loso_eval(X: np.ndarray, meta: pd.DataFrame, model_factory) -> Dict:
    persons = sorted(meta["person"].unique())
    fold_metrics = {}
    pred_rows: List[dict] = []
    for held in persons:
        train_mask = (meta["person"] != held).to_numpy()
        test_mask = (meta["person"] == held).to_numpy()
        if not train_mask.any() or not test_mask.any():
            continue
        model = model_factory()
        model.fit(X[train_mask], meta.loc[train_mask, "coarse_label"])
        y_pred = model.predict(X[test_mask])
        try:
            y_proba = model.predict_proba(X[test_mask])
            classes = list(model.classes_)
        except Exception:
            y_proba = None
            classes = list(CLASSES)
        y_true = meta.loc[test_mask, "coarse_label"].to_numpy()

        cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
        fold_metrics[held] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(
                y_true, y_pred, labels=list(CLASSES),
                average="macro", zero_division=0,
            )),
            "report": classification_report(
                y_true, y_pred, labels=list(CLASSES), digits=3, zero_division=0,
            ),
            "confusion": cm,
        }

        for i, vid in enumerate(meta.loc[test_mask, "video"].to_numpy()):
            row = {
                "video": vid,
                "person": held,
                "coarse_label": str(y_true[i]),
                "pred_aggregate": str(y_pred[i]),
                "fold": f"test={held}",
            }
            if y_proba is not None:
                for c, p in zip(classes, y_proba[i]):
                    row[f"prob_{c}"] = float(p)
            pred_rows.append(row)
    return {
        "fold_metrics": fold_metrics,
        "predictions": pd.DataFrame(pred_rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default=str(config.REPORTS_DIR / "fatigue_features.csv"),
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="FPS used for blink_rate aggregate (default 30, matches realtime).",
    )
    args = parser.parse_args(argv)

    config.ensure_dirs()
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: missing {feat_path}. Run extract_fatigue_features.py.")
        return 1
    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} face-frames from {feat_path}")

    X, meta = aggregate_features_from_dataframe(df, fps=args.fps)
    print(f"Built {X.shape[0]} clip vectors of dim {X.shape[1]} "
          f"({len(AGG_FEATURE_NAMES)} aggregate features).")

    summary_lines = ["=" * 72,
                     "FATIGUE — CLIP-AGGREGATE CLASSIFIER (LOSO)",
                     "=" * 72]
    summary_lines.append(f"Features: {list(AGG_FEATURE_NAMES)}")
    summary_lines.append(f"FPS for rate features: {args.fps}")
    summary_lines.append(f"Total clips: {len(meta)}")

    candidates = [
        ("LR (StandardScaler+LogReg)", _build_lr),
        ("RF (RandomForest)",          _build_rf),
    ]

    best_label = None
    best_mean_f1 = -1.0
    best_factory = None
    best_predictions = None
    for label, factory in candidates:
        result = _loso_eval(X, meta, factory)
        fold_metrics = result["fold_metrics"]
        accs = [m["accuracy"] for m in fold_metrics.values()]
        f1s = [m["macro_f1"] for m in fold_metrics.values()]
        mean_acc = float(np.mean(accs)) if accs else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        summary_lines.append("\n" + "-" * 72)
        summary_lines.append(f"Model: {label}")
        summary_lines.append("-" * 72)
        for person, m in fold_metrics.items():
            summary_lines.append(
                f"\n[test={person}]  accuracy={m['accuracy']:.3f}  "
                f"macro_F1={m['macro_f1']:.3f}"
            )
            summary_lines.append(m["report"])
            summary_lines.append(
                f"confusion (rows=true {CLASSES}, cols=pred):"
            )
            summary_lines.append(np.array2string(m["confusion"]))
        summary_lines.append(
            f"\n  Mean across folds: accuracy={mean_acc:.3f}  "
            f"macro_F1={mean_f1:.3f}"
        )
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_label = label
            best_factory = factory
            best_predictions = result["predictions"]
            best_fold_metrics = fold_metrics

    summary_lines.append("\n" + "=" * 72)
    summary_lines.append(
        f"Selected for deployment: {best_label}  "
        f"(LOSO mean macro-F1 = {best_mean_f1:.3f})"
    )
    summary_lines.append("=" * 72)

    # Save predictions for the headline summary.
    pred_csv = config.REPORTS_DIR / "fatigue_clip_eval_aggregate.csv"
    best_predictions.to_csv(pred_csv, index=False)
    print(f"Wrote {pred_csv}")

    # Render confusion matrices (per-fold + pooled) for the winner.
    pooled_cm = confusion_matrix(
        best_predictions["coarse_label"],
        best_predictions["pred_aggregate"],
        labels=list(CLASSES),
    )
    pooled_acc = float(accuracy_score(
        best_predictions["coarse_label"], best_predictions["pred_aggregate"],
    ))
    pooled_f1 = float(f1_score(
        best_predictions["coarse_label"], best_predictions["pred_aggregate"],
        labels=list(CLASSES), average="macro", zero_division=0,
    ))
    plot_confusion(
        cm=pooled_cm,
        classes=list(CLASSES),
        title=f"Fatigue per-clip — Aggregate-clf [pooled]  "
              f"(macro-F1={pooled_f1:.3f})",
        out_path=config.FIGURES_DIR
                 / "fatigue_clip_confusion_aggregate_pooled.png",
        normalize=False,
    )
    for person, m in best_fold_metrics.items():
        plot_confusion(
            cm=m["confusion"],
            classes=list(CLASSES),
            title=f"Fatigue per-clip — Aggregate-clf — test={person}  "
                  f"(macro-F1={m['macro_f1']:.3f})",
            out_path=config.FIGURES_DIR
                     / f"fatigue_clip_confusion_aggregate_test={person}.png",
            normalize=False,
        )

    # Refit best on ALL clips → deployment model (ships to realtime).
    deployment = best_factory()
    deployment.fit(X, meta["coarse_label"])
    classes_ = list(getattr(deployment, "classes_", CLASSES))
    payload = {
        "model": deployment,
        "feature_names": list(AGG_FEATURE_NAMES),
        "classes": classes_,
        "fps": float(args.fps),
        "loso_mean_macro_f1": best_mean_f1,
        "loso_label": best_label,
    }
    out_model = config.MODELS_DIR / "fatigue_aggregate.joblib"
    joblib.dump(payload, out_model)
    summary_lines.append(f"\nWrote deployment model: {out_model}")
    print(f"Wrote {out_model}")

    out_txt = config.REPORTS_DIR / "fatigue_aggregate_loso.txt"
    out_txt.write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"\nWrote {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
