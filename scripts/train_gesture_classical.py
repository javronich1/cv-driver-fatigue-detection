"""Stage 2C: train classical gesture classifiers (SVM + Random Forest).

Strategy:
  1. Leave-one-person-out cross-validation -> fold metrics.
  2. Final model fit on ALL data, saved to outputs/models/.

Outputs:
  outputs/models/gesture_svm.joblib
  outputs/models/gesture_rf.joblib
  outputs/reports/gesture_classical_eval.txt
  outputs/figures/gesture_confusion_<model>_<fold>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                            # noqa: E402
from src.gestures.classical import (                              # noqa: E402
    CLASSES, evaluate_loso, fit_on_all, make_random_forest, make_svm,
    save_model,
)


def plot_confusion(cm: np.ndarray, classes, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    config.ensure_dirs()
    feat_path = config.PROCESSED_DIR / "gestures" / "features.csv"
    if not feat_path.exists():
        print(f"ERROR: feature CSV not found at {feat_path}")
        print("       Run scripts/extract_gesture_features.py first.")
        return 1

    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} feature rows from {feat_path}")
    print(f"Persons: {sorted(df['person'].unique())}")
    print(f"Class counts:\n{df['label'].value_counts().to_string()}")
    print()

    report_lines = []

    for name, factory in [("SVM (RBF)", make_svm),
                          ("Random Forest", make_random_forest)]:
        print(f"=== {name} ===")
        report_lines.append(f"\n=== {name} ===")
        results = evaluate_loso(df, factory)
        accs, f1s = [], []
        for r in results:
            print(f"\nFold: {r.fold_name}")
            print(f"  accuracy : {r.accuracy:.3f}")
            print(f"  macro-F1 : {r.macro_f1:.3f}")
            print(r.report)
            print(f"  confusion ({CLASSES}):")
            print(r.confusion)
            accs.append(r.accuracy)
            f1s.append(r.macro_f1)

            report_lines.append(
                f"\nFold: {r.fold_name}\n"
                f"  accuracy: {r.accuracy:.3f}\n"
                f"  macro-F1: {r.macro_f1:.3f}\n"
                f"{r.report}"
                f"  confusion ({CLASSES}):\n  {r.confusion.tolist()}"
            )

            fig_path = (
                config.FIGURES_DIR
                / f"gesture_confusion_{name.split()[0].lower()}_{r.fold_name}.png"
            )
            plot_confusion(r.confusion, CLASSES,
                           f"{name} — {r.fold_name}", fig_path)

        mean_acc = float(np.mean(accs)) if accs else float("nan")
        mean_f1 = float(np.mean(f1s)) if f1s else float("nan")
        print(f"\n  Mean across folds: acc={mean_acc:.3f}  macro-F1={mean_f1:.3f}\n")
        report_lines.append(
            f"\n  Mean across folds: acc={mean_acc:.3f}  macro-F1={mean_f1:.3f}"
        )

        # Train final model on all data and save.
        final = fit_on_all(df, factory)
        out_name = "gesture_svm.joblib" if "SVM" in name else "gesture_rf.joblib"
        save_model(final, config.MODELS_DIR / out_name)
        print(f"  Saved final {name} model to {config.MODELS_DIR / out_name}")
        report_lines.append(
            f"  Saved final {name} model to "
            f"{config.MODELS_DIR / out_name}"
        )

    report_path = config.REPORTS_DIR / "gesture_classical_eval.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
