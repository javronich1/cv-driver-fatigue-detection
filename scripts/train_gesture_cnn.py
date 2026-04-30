"""Stage 3B: train the modern (CNN) gesture classifier with LOSO.

Inputs:
    data_processed/hand_crops/index.csv

Outputs:
    outputs/models/gesture_cnn.pt              (final model trained on all data)
    outputs/reports/gesture_cnn_loso.txt       (per-fold metrics)
    outputs/figures/gesture_confusion_cnn_test=person*.png
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
from src.gestures.cnn import (                                        # noqa: E402
    CLASSES, TrainConfig, best_device, evaluate_loso, fit_on_all,
    save_model,
)
from src.gestures.crops import CROP_INDEX                             # noqa: E402
from src.utils.plotting import plot_confusion                         # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-final", action="store_true",
                        help="Skip fitting the all-data deployment model.")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    if not CROP_INDEX.exists():
        print(f"ERROR: missing {CROP_INDEX}. Run extract_hand_crops.py first.")
        return 1

    df = pd.read_csv(CROP_INDEX)
    print(f"Loaded {len(df)} crops; classes:")
    print(df.groupby(["label", "person"]).size().unstack(fill_value=0))

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    device = best_device()
    print(f"\nDevice: {device}")

    # 1) LOSO evaluation
    results = evaluate_loso(df, cfg=cfg, device=device)

    # Persist textual report + confusion-matrix figures.
    report_lines = ["=" * 72,
                    "GESTURE CNN — LOSO RESULTS",
                    "=" * 72]
    macro_f1s = []
    accs = []
    for r in results:
        macro_f1s.append(r.macro_f1)
        accs.append(r.accuracy)
        report_lines.append(f"\n{r.fold_name}: accuracy={r.accuracy:.3f}  "
                            f"macro_F1={r.macro_f1:.3f}")
        report_lines.append(r.report)
        report_lines.append(f"confusion (rows=true {CLASSES}, cols=pred):")
        report_lines.append(np.array2string(r.confusion))

        plot_confusion(
            cm=r.confusion,
            classes=list(CLASSES),
            title=f"CNN — {r.fold_name}  (macro-F1={r.macro_f1:.3f})",
            out_path=config.FIGURES_DIR / f"gesture_confusion_cnn_{r.fold_name}.png",
            normalize=False,
        )

    if results:
        report_lines.append("\n" + "-" * 40)
        report_lines.append(f"Mean accuracy:  {np.mean(accs):.3f} "
                            f"(+/- {np.std(accs):.3f})")
        report_lines.append(f"Mean macro-F1:  {np.mean(macro_f1s):.3f} "
                            f"(+/- {np.std(macro_f1s):.3f})")
    report_lines.append("=" * 72)
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    out_txt = config.REPORTS_DIR / "gesture_cnn_loso.txt"
    out_txt.write_text(report_text)
    print(f"\nWrote {out_txt}")

    # 2) Fit on all data and save the deployment model.
    if not args.no_final:
        print("\n=== Final fit on all data ===")
        model = fit_on_all(df, cfg=cfg, device=device)
        out_model = config.MODELS_DIR / "gesture_cnn.pt"
        save_model(model, out_model)
        print(f"Saved {out_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
