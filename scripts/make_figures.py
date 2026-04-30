"""Generate report-ready figures from the inventory and gesture evaluation
artefacts (Stages 2 & 3).

Inputs (must already exist):
    outputs/reports/inventory.csv
    outputs/reports/gesture_sequence_eval_svm.csv
    outputs/reports/gesture_sequence_eval_cnn.csv      (optional)

Outputs (PNG, written to outputs/figures/):
    dataset_class_counts_gestures.png
    dataset_class_counts_fatigue.png
    dataset_duration_per_class.png
    gesture_classical_loso_macro_f1.png
    gesture_loso_macro_f1_classical_vs_cnn.png         (Stage 3D)
    gesture_sequence_per_folder_accuracy.png
    gesture_sequence_activation_confusion.png
    gesture_sequence_activation_confusion_norm.png
    gesture_sequence_per_folder_accuracy_cnn.png       (Stage 3D)
    gesture_sequence_activation_confusion_cnn.png      (Stage 3D)
    gesture_sequence_activation_f1_classical_vs_cnn.png(Stage 3D)
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
    # Hard-coded from latest training runs.
    "SVM (RBF)":         {"person1": 0.813, "person2": 0.800},
    "Random Forest":     {"person1": 0.865, "person2": 0.750},
    # Stage 3B: MobileNetV3-Small fine-tuned on 96x96 hand crops.
    "CNN (MobileNetV3)": {"person1": 0.695, "person2": 0.702},
}


def figure_loso_macro_f1() -> None:
    folds = ["person1", "person2"]

    # Classical-only chart (kept for backwards compatibility).
    classical = {k: v for k, v in LOSO_RESULTS.items()
                 if k in ("SVM (RBF)", "Random Forest")}
    plot_grouped_bars(
        categories=[f"test={f}" for f in folds],
        series={m: [LOSO_RESULTS[m][f] for f in folds] for m in classical},
        title="Per-frame gesture classifier — LOSO macro-F1",
        ylabel="macro-F1",
        out_path=config.FIGURES_DIR / "gesture_classical_loso_macro_f1.png",
        ylim=(0, 1),
        figsize=(7, 4),
    )

    # Classical vs modern (Stage 3D).
    plot_grouped_bars(
        categories=[f"test={f}" for f in folds],
        series={m: [LOSO_RESULTS[m][f] for f in folds] for m in LOSO_RESULTS},
        title="Per-frame gesture classifier — classical vs modern (LOSO macro-F1)",
        ylabel="macro-F1",
        out_path=config.FIGURES_DIR
                 / "gesture_loso_macro_f1_classical_vs_cnn.png",
        ylim=(0, 1),
        figsize=(8.5, 4.5),
    )


# ---------------------------------------------------------------------------
# 3. Sequence-level (end-to-end) figures
# ---------------------------------------------------------------------------
def _activation_metrics(seq: pd.DataFrame) -> Dict[str, float]:
    pos = seq["should_activate"].astype(bool)
    pred = seq["activated"].astype(bool)
    tp = int((pos & pred).sum())
    fp = int((~pos & pred).sum())
    fn = int((pos & ~pred).sum())
    tn = int((~pos & ~pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": float(seq["correct"].mean()) if len(seq) else 0.0,
    }


def figures_from_sequence_eval(
    seq: pd.DataFrame,
    *,
    model_label: str,
    file_suffix: str = "",
) -> Dict[str, float]:
    """Render per-folder accuracy + activation confusion matrices.

    ``model_label`` is shown in the figure titles; ``file_suffix`` is appended
    to the output filenames so multiple models can coexist on disk.
    Returns the activation metrics dict so the caller can build a comparison
    chart.
    """
    sfx = f"_{file_suffix}" if file_suffix else ""

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

    plot_horizontal_bars(
        labels=[
            f"{row['folder_label']}  (n={int(row['n'])}, "
            f"GT={'activate' if row['should_activate'] else 'reject'})"
            for _, row in folder_summary.iterrows()
        ],
        values=folder_summary["accuracy"].tolist(),
        title=f"End-to-end activation accuracy by clip type — {model_label}",
        xlabel="accuracy",
        out_path=config.FIGURES_DIR
                 / f"gesture_sequence_per_folder_accuracy{sfx}.png",
        figsize=(9, 5),
    )

    # 3b. Binary activation confusion matrix.
    m = _activation_metrics(seq)
    cm = np.array([[m["tp"], m["fn"]],
                   [m["fp"], m["tn"]]])
    classes = ["activate", "reject"]
    plot_confusion(
        cm=cm,
        classes=classes,
        title=f"Activation outcome — {model_label}  "
              f"(P={m['precision']:.2f}, R={m['recall']:.2f})",
        out_path=config.FIGURES_DIR
                 / f"gesture_sequence_activation_confusion{sfx}.png",
        normalize=False,
    )
    plot_confusion(
        cm=cm,
        classes=classes,
        title=f"Activation outcome (row-normalised) — {model_label}",
        out_path=config.FIGURES_DIR
                 / f"gesture_sequence_activation_confusion_norm{sfx}.png",
        normalize=True,
    )
    return m


def figure_activation_comparison(metrics: Dict[str, Dict[str, float]]) -> None:
    """Side-by-side bars: precision / recall / F1 / accuracy per model."""
    if not metrics:
        return
    metric_keys = ("precision", "recall", "f1", "accuracy")
    series: Dict[str, list] = {}
    for model_name, m in metrics.items():
        series[model_name] = [m[k] for k in metric_keys]
    plot_grouped_bars(
        categories=list(metric_keys),
        series=series,
        title="End-to-end gesture activation — classical vs modern",
        ylabel="score",
        out_path=config.FIGURES_DIR
                 / "gesture_sequence_activation_f1_classical_vs_cnn.png",
        ylim=(0, 1),
        figsize=(8.5, 4.5),
    )


def main() -> int:
    config.ensure_dirs()

    inv_csv = config.REPORTS_DIR / "inventory.csv"
    seq_svm = config.REPORTS_DIR / "gesture_sequence_eval_svm.csv"
    seq_cnn = config.REPORTS_DIR / "gesture_sequence_eval_cnn.csv"
    if not inv_csv.exists():
        print(f"ERROR: missing {inv_csv}. Run scripts/inventory.py.")
        return 1
    if not seq_svm.exists():
        print(f"ERROR: missing {seq_svm}. "
              f"Run scripts/eval_gesture_sequence.py.")
        return 1

    inv = pd.read_csv(inv_csv)
    figures_from_inventory(inv)
    figure_loso_macro_f1()

    activation_metrics: Dict[str, Dict[str, float]] = {}
    activation_metrics["SVM"] = figures_from_sequence_eval(
        pd.read_csv(seq_svm),
        model_label="SVM + state machine",
        file_suffix="",  # keep historical filenames
    )
    if seq_cnn.exists():
        activation_metrics["CNN"] = figures_from_sequence_eval(
            pd.read_csv(seq_cnn),
            model_label="CNN + state machine",
            file_suffix="cnn",
        )
        figure_activation_comparison(activation_metrics)
    else:
        print(f"NOTE: {seq_cnn} not found — skipping CNN comparison figures.")

    print("Figures written to:", config.FIGURES_DIR)
    for p in sorted(config.FIGURES_DIR.glob("*.png")):
        print(" ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
