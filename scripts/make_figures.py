"""Generate report-ready figures from the inventory and Stage-2 evaluation
artefacts.

Inputs (must already exist):
    outputs/reports/inventory.csv
    outputs/reports/gesture_sequence_eval_svm.csv

Outputs (PNG, written to outputs/figures/):
    dataset_class_counts_gestures.png
    dataset_class_counts_fatigue.png
    dataset_duration_per_class.png
    gesture_classical_loso_macro_f1.png
    gesture_sequence_per_folder_accuracy.png
    gesture_sequence_activation_confusion.png
    gesture_sequence_activation_confusion_norm.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                         # noqa: E402
from src.utils.plotting import (                               # noqa: E402
    plot_confusion, plot_grouped_bars, plot_horizontal_bars,
)


# ---------------------------------------------------------------------------
# 1. Dataset class distributions (from inventory)
# ---------------------------------------------------------------------------
def figures_from_inventory(inv: pd.DataFrame) -> None:
    for ds in ("gestures", "fatigue"):
        sub = inv[inv["dataset"] == ds]
        if sub.empty:
            continue
        counts = (sub.groupby(["canonical_label", "person"]).size()
                     .unstack(fill_value=0).sort_index())
        # Per-class clip counts split by person
        plot_grouped_bars(
            categories=list(counts.index),
            series={p: counts[p].tolist() for p in counts.columns},
            title=f"Clips per class — {ds}",
            ylabel="number of clips",
            out_path=config.FIGURES_DIR / f"dataset_class_counts_{ds}.png",
            figsize=(11, 4.5) if ds == "fatigue" else (9, 4.5),
            annotate=True,
        )

    # Total duration per class (gestures + fatigue, separated by dataset)
    dur = (inv.groupby(["dataset", "canonical_label"])["duration_s"]
              .sum().reset_index())
    dur["minutes"] = dur["duration_s"] / 60.0
    dur = dur.sort_values(["dataset", "minutes"], ascending=[True, False])
    labels = [f"[{r['dataset']}] {r['canonical_label']}" for _, r in dur.iterrows()]
    plot_horizontal_bars(
        labels=labels,
        values=dur["minutes"].tolist(),
        title="Total duration per class (minutes)",
        xlabel="minutes",
        out_path=config.FIGURES_DIR / "dataset_duration_per_class.png",
        figsize=(9, 9),
    )


# ---------------------------------------------------------------------------
# 2. Classical LOSO macro-F1 comparison (parsed from training output)
# ---------------------------------------------------------------------------
LOSO_RESULTS = {
    # Hard-coded from scripts/train_gesture_classical.py latest run.
    "SVM (RBF)":     {"person1": 0.813, "person2": 0.800},
    "Random Forest": {"person1": 0.865, "person2": 0.750},
}


def figure_loso_macro_f1() -> None:
    folds = ["person1", "person2"]
    series: Dict[str, list] = {}
    for model, vals in LOSO_RESULTS.items():
        series[model] = [vals[f] for f in folds]
    plot_grouped_bars(
        categories=[f"test={f}" for f in folds],
        series=series,
        title="Per-frame gesture classifier — LOSO macro-F1",
        ylabel="macro-F1",
        out_path=config.FIGURES_DIR / "gesture_classical_loso_macro_f1.png",
        ylim=(0, 1),
        figsize=(7, 4),
    )


# ---------------------------------------------------------------------------
# 3. Sequence-level (end-to-end) figures
# ---------------------------------------------------------------------------
def figures_from_sequence_eval(seq: pd.DataFrame) -> None:
    # 3a. Per-folder accuracy bar chart with desired-vs-actual activations.
    folder_summary = (
        seq.groupby("folder_label").agg(
            n=("activated", "size"),
            n_activated=("activated", "sum"),
            should_activate=("should_activate", "first"),
            accuracy=("correct", "mean"),
        ).reset_index()
        .sort_values("folder_label")
    )

    # Accuracy per folder
    plot_horizontal_bars(
        labels=[
            f"{row['folder_label']}  (n={int(row['n'])}, "
            f"GT={'activate' if row['should_activate'] else 'reject'})"
            for _, row in folder_summary.iterrows()
        ],
        values=folder_summary["accuracy"].tolist(),
        title="End-to-end activation accuracy by clip type (SVM + state machine)",
        xlabel="accuracy",
        out_path=config.FIGURES_DIR / "gesture_sequence_per_folder_accuracy.png",
        figsize=(9, 5),
    )

    # 3b. Binary activation confusion matrix.
    pos = seq["should_activate"].astype(bool)
    pred = seq["activated"].astype(bool)
    tp = int(((pos) & (pred)).sum())
    fp = int(((~pos) & (pred)).sum())
    fn = int(((pos) & (~pred)).sum())
    tn = int(((~pos) & (~pred)).sum())
    cm = np.array([[tp, fn], [fp, tn]])  # rows: should_activate yes/no, cols: predicted yes/no
    classes = ["activate", "reject"]
    plot_confusion(
        cm=cm,
        classes=classes,
        title=f"Activation outcome (P={tp/(tp+fp):.2f}, R={tp/(tp+fn):.2f})",
        out_path=config.FIGURES_DIR / "gesture_sequence_activation_confusion.png",
        normalize=False,
    )
    plot_confusion(
        cm=cm,
        classes=classes,
        title="Activation outcome — row-normalised",
        out_path=config.FIGURES_DIR / "gesture_sequence_activation_confusion_norm.png",
        normalize=True,
    )


def main() -> int:
    config.ensure_dirs()

    inv_csv = config.REPORTS_DIR / "inventory.csv"
    seq_csv = config.REPORTS_DIR / "gesture_sequence_eval_svm.csv"
    if not inv_csv.exists():
        print(f"ERROR: missing {inv_csv}. Run scripts/inventory.py.")
        return 1
    if not seq_csv.exists():
        print(f"ERROR: missing {seq_csv}. Run scripts/eval_gesture_sequence.py.")
        return 1

    inv = pd.read_csv(inv_csv)
    seq = pd.read_csv(seq_csv)

    figures_from_inventory(inv)
    figure_loso_macro_f1()
    figures_from_sequence_eval(seq)

    print("Figures written to:", config.FIGURES_DIR)
    for p in sorted(config.FIGURES_DIR.glob("*.png")):
        print(" ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
