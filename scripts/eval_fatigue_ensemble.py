"""Stage 5F: per-clip eval of the ensemble (heuristic + aggregate-clf).

We evaluate the deployment ensemble used by ``run_realtime.py`` on the
exact same per-clip LOSO setup as every other fatigue model in the
report. The ensemble is a 50/50 probability average of:

    * The heuristic predictor (Stage 5D — data-free MediaPipe rule).
    * The clip-aggregate classifier (Stage 5E — trained on 15 LOSO
      summary statistics per clip).

Inputs:
    outputs/reports/fatigue_features.csv
    outputs/models/fatigue_aggregate.joblib   (Stage 5E artefact)

Outputs:
    outputs/reports/fatigue_clip_eval_ensemble.csv
    outputs/reports/fatigue_ensemble_clip_eval.txt
    outputs/figures/fatigue_clip_confusion_ensemble_pooled.png
    outputs/figures/fatigue_clip_confusion_ensemble_test=person*.png

Notes
-----
Per-fold LOSO faithfulness: the aggregate classifier is **retrained per
fold** on the held-in person, so this CSV reflects a true LOSO ensemble
estimate (we don't reuse the deployment all-data model). The deployment
``.joblib`` is fit on all clips for production use, as is standard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                  # noqa: E402
from src.fatigue.aggregate import (                                     # noqa: E402
    AGG_FEATURE_NAMES, aggregate_features, aggregate_features_from_dataframe,
)
from src.fatigue.classical import CLASSES                               # noqa: E402
from src.fatigue.features import FEATURE_NAMES                          # noqa: E402
from src.system.realtime import (                                       # noqa: E402
    make_heuristic_predictor,
)
from src.utils.plotting import plot_confusion                           # noqa: E402


def _build_aggregate_rf() -> RandomForestClassifier:
    """Same LOSO-winning estimator as scripts/train_fatigue_aggregate.py."""
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default=str(config.REPORTS_DIR / "fatigue_features.csv"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--heuristic-weight", type=float, default=0.5,
                        help="Probability-average weight for the heuristic "
                             "(only used when --strategy=blend).")
    parser.add_argument("--strategy",
                        choices=("blend", "yawn_route"),
                        default="yawn_route",
                        help="blend = weighted probability average; "
                             "yawn_route = aggregate-clf default, "
                             "yield to heuristic when its top class is "
                             "'yawning' (Stage 5F default).")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: missing {feat_path}. Run extract_fatigue_features.py.")
        return 1
    df = pd.read_csv(feat_path)

    # Build all clip vectors once.
    X_agg, meta = aggregate_features_from_dataframe(df, fps=args.fps)
    feature_cols = list(FEATURE_NAMES)

    heuristic = make_heuristic_predictor()
    classes = list(CLASSES)
    h_w = float(args.heuristic_weight)
    a_w = 1.0 - h_w

    summary_lines = ["=" * 72,
                     "FATIGUE — ENSEMBLE (heuristic + aggregate-clf)",
                     "=" * 72]
    summary_lines.append(f"Strategy: {args.strategy}")
    if args.strategy == "blend":
        summary_lines.append(
            f"Probability average — heuristic={h_w:.2f}  aggregate={a_w:.2f}"
        )
    else:
        summary_lines.append(
            "yawn_route: aggregate-clf default; heuristic vetoes when its "
            "top class is 'yawning' (heuristic's strongest signal — beats "
            "aggregate on yawning recall on the harder fold)."
        )
    summary_lines.append(f"Total clips: {len(meta)}")

    persons = sorted(meta["person"].unique())
    pred_rows: List[dict] = []
    fold_metrics = {}

    for held in persons:
        train_mask = (meta["person"] != held).to_numpy()
        test_mask = (meta["person"] == held).to_numpy()
        if not train_mask.any() or not test_mask.any():
            continue
        # Train aggregate-clf on the held-in person only (true LOSO).
        rf = _build_aggregate_rf()
        rf.fit(X_agg[train_mask], meta.loc[train_mask, "coarse_label"])
        rf_classes = list(rf.classes_)

        y_true_list, y_pred_list = [], []
        for i in np.where(test_mask)[0]:
            video = meta.loc[i, "video"]
            sub = df[df["video"] == video]
            buf = sub[feature_cols].to_numpy(dtype=np.float32)
            # Heuristic on raw buffer.
            h_label, _, h_probs = heuristic(buf)
            # Aggregate classifier on aggregate features.
            agg = aggregate_features(buf, fps=args.fps).reshape(1, -1)
            a_probs_arr = rf.predict_proba(agg)[0]
            a_probs = {c: float(p) for c, p in zip(rf_classes, a_probs_arr)}
            all_classes = set(h_probs) | set(a_probs)
            if args.strategy == "yawn_route":
                # Aggregate is the default voice. Heuristic vetoes only
                # when its top class is "yawning" (its strongest signal).
                if h_label == "yawning":
                    blended = h_probs
                else:
                    blended = {c: float(a_probs.get(c, 0.0))
                               for c in all_classes}
            else:
                # Weighted probability average across the union of classes.
                blended = {
                    c: h_w * float(h_probs.get(c, 0.0))
                       + a_w * float(a_probs.get(c, 0.0))
                    for c in all_classes
                }
            label = max(blended, key=blended.get)
            row = {
                "video": video,
                "person": held,
                "coarse_label": str(meta.loc[i, "coarse_label"]),
                "pred_ensemble": label,
                "fold": f"test={held}",
            }
            for c in classes:
                row[f"prob_{c}"] = float(blended.get(c, 0.0))
            pred_rows.append(row)
            y_true_list.append(row["coarse_label"])
            y_pred_list.append(label)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        rep = classification_report(
            y_true, y_pred, labels=classes, digits=3, zero_division=0,
        )
        fold_metrics[held] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(
                y_true, y_pred, labels=classes, average="macro", zero_division=0,
            )),
            "report": rep,
            "confusion": cm,
        }
        summary_lines.append("\n" + "-" * 72)
        summary_lines.append(
            f"[test={held}]  accuracy={fold_metrics[held]['accuracy']:.3f}  "
            f"macro_F1={fold_metrics[held]['macro_f1']:.3f}"
        )
        summary_lines.append("-" * 72)
        summary_lines.append(rep)
        summary_lines.append(f"confusion (rows=true {classes}, cols=pred):")
        summary_lines.append(np.array2string(cm))
        plot_confusion(
            cm=cm,
            classes=classes,
            title=f"Fatigue ensemble — test={held}  "
                  f"(macro-F1={fold_metrics[held]['macro_f1']:.3f})",
            out_path=config.FIGURES_DIR
                     / f"fatigue_clip_confusion_ensemble_test={held}.png",
            normalize=False,
        )

    # Pooled.
    df_pred = pd.DataFrame(pred_rows)
    pooled_cm = confusion_matrix(
        df_pred["coarse_label"], df_pred["pred_ensemble"], labels=classes,
    )
    pooled_acc = float(accuracy_score(
        df_pred["coarse_label"], df_pred["pred_ensemble"],
    ))
    pooled_f1 = float(f1_score(
        df_pred["coarse_label"], df_pred["pred_ensemble"],
        labels=classes, average="macro", zero_division=0,
    ))
    summary_lines.append("\n" + "=" * 72)
    summary_lines.append(
        f"Pooled  accuracy={pooled_acc:.3f}  macro_F1={pooled_f1:.3f}"
    )
    summary_lines.append("=" * 72)
    summary_lines.append(np.array2string(pooled_cm))
    summary_lines.append(
        f"\nPer-person mean accuracy="
        f"{np.mean([m['accuracy'] for m in fold_metrics.values()]):.3f}  "
        f"macro_F1="
        f"{np.mean([m['macro_f1'] for m in fold_metrics.values()]):.3f}"
    )
    plot_confusion(
        cm=pooled_cm,
        classes=classes,
        title=f"Fatigue ensemble — pooled  (macro-F1={pooled_f1:.3f})",
        out_path=config.FIGURES_DIR
                 / "fatigue_clip_confusion_ensemble_pooled.png",
        normalize=False,
    )

    out_csv = config.REPORTS_DIR / "fatigue_clip_eval_ensemble.csv"
    df_pred.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_txt = config.REPORTS_DIR / "fatigue_ensemble_clip_eval.txt"
    out_txt.write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"\nWrote {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
