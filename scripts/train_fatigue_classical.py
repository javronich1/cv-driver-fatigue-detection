"""Stage 4C: train + LOSO-evaluate the classical fatigue classifier.

Inputs:
    outputs/reports/fatigue_features.csv

Outputs:
    outputs/models/fatigue_svm.joblib
    outputs/models/fatigue_rf.joblib
    outputs/reports/fatigue_classical_loso.txt
    outputs/figures/fatigue_confusion_<svm|random>_test=person*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                # noqa: E402
from src.fatigue.classical import (                                   # noqa: E402
    CLASSES, evaluate_loso, fit_on_all, make_random_forest, make_svm,
    save_model,
)
from src.utils.plotting import plot_confusion                         # noqa: E402


def _print_and_collect(name: str, results, lines):
    lines.append("\n" + "=" * 72)
    lines.append(f"{name.upper()} — LOSO RESULTS")
    lines.append("=" * 72)
    accs, f1s = [], []
    for r in results:
        accs.append(r.accuracy)
        f1s.append(r.macro_f1)
        lines.append(f"\n{r.fold_name}: accuracy={r.accuracy:.3f}  "
                     f"macro_F1={r.macro_f1:.3f}")
        lines.append(r.report)
        lines.append(f"confusion (rows=true {CLASSES}, cols=pred):")
        lines.append(np.array2string(r.confusion))
    if results:
        lines.append("-" * 40)
        lines.append(f"Mean accuracy:  {np.mean(accs):.3f} "
                     f"(+/- {np.std(accs):.3f})")
        lines.append(f"Mean macro-F1:  {np.mean(f1s):.3f} "
                     f"(+/- {np.std(f1s):.3f})")
    return accs, f1s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",
                        default=str(config.REPORTS_DIR / "fatigue_features.csv"))
    parser.add_argument("--no-final", action="store_true",
                        help="Skip fitting the all-data deployment models.")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: features CSV not found at {feat_path}.")
        print("       Run scripts/extract_fatigue_features.py first.")
        return 1
    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} face-frames from {feat_path}")
    print("Class distribution:")
    print(df.groupby(["coarse_label", "person"]).size().unstack(fill_value=0))

    lines = []
    # SVM
    print("\nTraining SVM (RBF) under LOSO ...")
    svm_results = evaluate_loso(df, make_svm)
    _print_and_collect("SVM (RBF)", svm_results, lines)

    # RF
    print("\nTraining Random Forest under LOSO ...")
    rf_results = evaluate_loso(df, make_random_forest)
    _print_and_collect("Random Forest", rf_results, lines)

    # Confusion-matrix figures.
    for name_short, results in (("svm", svm_results), ("random", rf_results)):
        for r in results:
            plot_confusion(
                cm=r.confusion,
                classes=list(CLASSES),
                title=f"Fatigue {name_short} — {r.fold_name}  "
                      f"(macro-F1={r.macro_f1:.3f})",
                out_path=config.FIGURES_DIR
                         / f"fatigue_confusion_{name_short}_{r.fold_name}.png",
                normalize=False,
            )

    out_txt = config.REPORTS_DIR / "fatigue_classical_loso.txt"
    out_txt.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_txt}")

    # Deployment models on all data.
    if not args.no_final:
        print("\nFitting deployment models on all data ...")
        svm = fit_on_all(df, make_svm)
        rf = fit_on_all(df, make_random_forest)
        save_model(svm, config.MODELS_DIR / "fatigue_svm.joblib")
        save_model(rf,  config.MODELS_DIR / "fatigue_rf.joblib")
        print(f"Saved {config.MODELS_DIR / 'fatigue_svm.joblib'}")
        print(f"Saved {config.MODELS_DIR / 'fatigue_rf.joblib'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
