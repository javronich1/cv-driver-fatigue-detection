"""Stage 5D: per-clip evaluation of the heuristic (rule-based) fatigue scorer.

The heuristic predictor is data-free — it scores every face-frame from
MediaPipe's face-relative blendshapes (``eyeBlinkLeft/Right`` and
``jawOpen``). It exists because the trained models (classical + temporal
CNN) overfit our 2 training subjects and degrade on unseen people; the
heuristic generalises by construction.

This script applies the same rule used in the realtime system to every
face-frame in ``fatigue_features.csv`` and aggregates each clip with the
identical "buffer mean" decision (whole-clip mean blink and jaw),
producing per-person and pooled per-clip metrics directly comparable to
the SVM / RF / Temporal-CNN tables.

Inputs:
    outputs/reports/fatigue_features.csv

Outputs:
    outputs/reports/fatigue_clip_eval_heuristic.csv
    outputs/reports/fatigue_heuristic_clip_eval.txt
    outputs/figures/fatigue_clip_confusion_heuristic_pooled.png
    outputs/figures/fatigue_clip_confusion_heuristic_test=person*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                          # noqa: E402
from src.fatigue.classical import CLASSES                       # noqa: E402
from src.utils.plotting import plot_confusion                   # noqa: E402


# Same defaults as :func:`make_heuristic_predictor` in src/system/realtime.py.
DEFAULT_DROWSY_BLINK_THRESHOLD = 0.45
DEFAULT_YAWN_JAW_THRESHOLD = 0.35


def heuristic_clip_label(
    blink_mean: float,
    jaw_mean: float,
    *,
    drowsy_blink_threshold: float,
    yawn_jaw_threshold: float,
) -> str:
    """Mirror of ``make_heuristic_predictor`` but on whole-clip means.

    Continuous, bounded "evidence" for each class; argmax decides.
    """
    drowsy_sig = min(1.0, max(0.0, blink_mean / max(drowsy_blink_threshold, 1e-3)))
    yawn_sig = min(1.0, max(0.0, jaw_mean / max(yawn_jaw_threshold, 1e-3)))
    alert_sig = max(0.0, 1.0 - max(drowsy_sig, yawn_sig))
    scores = {"alert": alert_sig, "drowsy": drowsy_sig, "yawning": yawn_sig}
    return max(scores, key=scores.get)


def predict_clips(
    df: pd.DataFrame,
    *,
    drowsy_blink_threshold: float,
    yawn_jaw_threshold: float,
) -> pd.DataFrame:
    """One row per clip with the heuristic prediction."""
    rows: List[dict] = []
    for video, sub in df.groupby("video", sort=False):
        blink_mean = float(
            (sub["eyeBlinkLeft"].to_numpy() + sub["eyeBlinkRight"].to_numpy()).mean() / 2
        )
        jaw_mean = float(sub["jawOpen"].mean())
        pred = heuristic_clip_label(
            blink_mean, jaw_mean,
            drowsy_blink_threshold=drowsy_blink_threshold,
            yawn_jaw_threshold=yawn_jaw_threshold,
        )
        rows.append(dict(
            video=video,
            person=str(sub["person"].iloc[0]),
            folder_label=str(sub["folder_label"].iloc[0]),
            fine_label=str(sub["fine_label"].iloc[0]),
            coarse_label=str(sub["coarse_label"].iloc[0]),
            n_frames=int(len(sub)),
            blink_mean=blink_mean,
            jaw_mean=jaw_mean,
            pred_heuristic=pred,
        ))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default=str(config.REPORTS_DIR / "fatigue_features.csv"),
    )
    parser.add_argument(
        "--drowsy-blink-threshold", type=float,
        default=DEFAULT_DROWSY_BLINK_THRESHOLD,
    )
    parser.add_argument(
        "--yawn-jaw-threshold", type=float,
        default=DEFAULT_YAWN_JAW_THRESHOLD,
    )
    args = parser.parse_args(argv)

    config.ensure_dirs()
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: missing {feat_path}. Run extract_fatigue_features.py.")
        return 1
    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} face-frames from {feat_path}")

    clip_df = predict_clips(
        df,
        drowsy_blink_threshold=args.drowsy_blink_threshold,
        yawn_jaw_threshold=args.yawn_jaw_threshold,
    )

    # Per-person and pooled metrics.
    summary_lines = ["=" * 72, "FATIGUE — HEURISTIC PER-CLIP EVAL", "=" * 72]
    summary_lines.append(
        f"drowsy_blink_threshold = {args.drowsy_blink_threshold}"
    )
    summary_lines.append(
        f"yawn_jaw_threshold     = {args.yawn_jaw_threshold}"
    )
    summary_lines.append(f"Total clips            : {len(clip_df)}")

    per_person_acc: List[float] = []
    per_person_f1: List[float] = []
    persons = sorted(clip_df["person"].unique())
    fold_records: List[pd.DataFrame] = []

    for person in persons:
        sub = clip_df[clip_df["person"] == person]
        y_true = sub["coarse_label"].to_numpy()
        y_pred = sub["pred_heuristic"].to_numpy()
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(
            y_true, y_pred, labels=list(CLASSES),
            average="macro", zero_division=0,
        ))
        per_person_acc.append(acc)
        per_person_f1.append(f1)
        cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
        rep = classification_report(
            y_true, y_pred, labels=list(CLASSES), digits=3, zero_division=0,
        )
        summary_lines.append("\n" + "-" * 72)
        summary_lines.append(
            f"[test={person}]  accuracy={acc:.3f}  macro_F1={f1:.3f}"
        )
        summary_lines.append("-" * 72)
        summary_lines.append(rep)
        summary_lines.append(
            f"confusion (rows=true {CLASSES}, cols=pred):"
        )
        summary_lines.append(np.array2string(cm))
        plot_confusion(
            cm=cm,
            classes=list(CLASSES),
            title=f"Fatigue heuristic — test={person}  "
                  f"(macro-F1={f1:.3f})",
            out_path=config.FIGURES_DIR
                     / f"fatigue_clip_confusion_heuristic_test={person}.png",
            normalize=False,
        )
        fold_df = sub.copy()
        fold_df["fold"] = f"test={person}"
        fold_records.append(fold_df)

    # Pooled (all clips, all persons).
    y_true = clip_df["coarse_label"].to_numpy()
    y_pred = clip_df["pred_heuristic"].to_numpy()
    pooled_acc = float(accuracy_score(y_true, y_pred))
    pooled_f1 = float(f1_score(
        y_true, y_pred, labels=list(CLASSES),
        average="macro", zero_division=0,
    ))
    pooled_cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    summary_lines.append("\n" + "=" * 72)
    summary_lines.append(
        f"Pooled  accuracy={pooled_acc:.3f}  macro_F1={pooled_f1:.3f}"
    )
    summary_lines.append("=" * 72)
    summary_lines.append(np.array2string(pooled_cm))
    summary_lines.append(
        f"\nPer-person mean accuracy={np.mean(per_person_acc):.3f}  "
        f"macro_F1={np.mean(per_person_f1):.3f}"
    )
    plot_confusion(
        cm=pooled_cm,
        classes=list(CLASSES),
        title=f"Fatigue heuristic — pooled  "
              f"(macro-F1={pooled_f1:.3f})",
        out_path=config.FIGURES_DIR
                 / "fatigue_clip_confusion_heuristic_pooled.png",
        normalize=False,
    )

    # Persist per-clip predictions (with fold col) and the text summary.
    out_csv = config.REPORTS_DIR / "fatigue_clip_eval_heuristic.csv"
    pd.concat(fold_records, ignore_index=True).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_txt = config.REPORTS_DIR / "fatigue_heuristic_clip_eval.txt"
    out_txt.write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"\nWrote {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
