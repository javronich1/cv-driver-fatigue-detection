"""Stage 5: train + LOSO-evaluate the modern (1D temporal-CNN) fatigue
classifier directly on per-frame feature *sequences*.

Inputs:
    outputs/reports/fatigue_features.csv

Outputs:
    outputs/models/fatigue_temporal_cnn.pt
    outputs/reports/fatigue_temporal_cnn_loso.txt
    outputs/reports/fatigue_temporal_cnn_clip_eval.csv
    outputs/figures/fatigue_temporal_cnn_confusion_test=person*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                  # noqa: E402
from src.fatigue.temporal_cnn import (                                  # noqa: E402
    CLASSES, SEQ_LEN, best_device, evaluate_loso, fit_on_all, save_model,
)
from src.utils.plotting import plot_confusion                           # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",
                        default=str(config.REPORTS_DIR / "fatigue_features.csv"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--no-final", action="store_true",
                        help="Skip refitting on all data.")
    # Augmentation (defaults match the post-tuning recipe — see README).
    parser.add_argument("--augment-noise-std", type=float, default=0.05,
                        help="Gaussian noise std on standardised features "
                             "(real frames only; train-only). Default 0.05.")
    parser.add_argument("--augment-time-shift", type=int, default=4,
                        help="Random temporal crop shift in frames "
                             "(train-only). Default 4.")
    parser.add_argument("--augment-feature-dropout", type=float, default=0.05,
                        help="Probability of zeroing a feature channel for "
                             "the whole clip (train-only). Default 0.05.")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: missing {feat_path}. Run extract_fatigue_features.py.")
        return 1

    df = pd.read_csv(feat_path)
    n_clips = df["video"].nunique()
    print(f"Loaded {len(df)} face-frames spanning {n_clips} clips "
          f"({df['person'].nunique()} persons).")
    print(f"Device: {best_device()}    seq_len={args.seq_len}    "
          f"epochs={args.epochs}    batch_size={args.batch_size}")

    print(f"Augmentation: noise_std={args.augment_noise_std}  "
          f"time_shift=+/-{args.augment_time_shift}  "
          f"feature_dropout_p={args.augment_feature_dropout}")

    results, clip_preds = evaluate_loso(
        df,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment_noise_std=args.augment_noise_std,
        augment_time_shift=args.augment_time_shift,
        augment_feature_dropout=args.augment_feature_dropout,
    )

    # ---- Report ----
    lines = []
    lines.append("=" * 72)
    lines.append("FATIGUE TEMPORAL-CNN — LOSO RESULTS")
    lines.append("=" * 72)
    lines.append(f"seq_len={args.seq_len}  epochs={args.epochs}  "
                 f"batch_size={args.batch_size}  lr={args.lr}")
    accs, f1s = [], []
    for r in results:
        accs.append(r.accuracy)
        f1s.append(r.macro_f1)
        lines.append("\n" + "-" * 72)
        lines.append(f"{r.fold_name}: accuracy={r.accuracy:.3f}  "
                     f"macro_F1={r.macro_f1:.3f}")
        lines.append(r.report)
        lines.append(f"confusion (rows=true {CLASSES}, cols=pred):")
        lines.append(np.array2string(r.confusion))

        plot_confusion(
            cm=r.confusion,
            classes=list(CLASSES),
            title=f"Fatigue temporal-CNN — {r.fold_name}  "
                  f"(macro-F1={r.macro_f1:.3f})",
            out_path=config.FIGURES_DIR
                     / f"fatigue_temporal_cnn_confusion_{r.fold_name}.png",
            normalize=False,
        )

    if results:
        lines.append("-" * 72)
        lines.append(f"Mean accuracy:  {np.mean(accs):.3f} "
                     f"(+/- {np.std(accs):.3f})")
        lines.append(f"Mean macro-F1:  {np.mean(f1s):.3f} "
                     f"(+/- {np.std(f1s):.3f})")

    out_txt = config.REPORTS_DIR / "fatigue_temporal_cnn_loso.txt"
    out_txt.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_txt}")

    out_csv = config.REPORTS_DIR / "fatigue_temporal_cnn_clip_eval.csv"
    clip_preds.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # ---- Final deployment model on all data ----
    if not args.no_final:
        print("\nFitting deployment model on all data ...")
        model, mean, std = fit_on_all(
            df, seq_len=args.seq_len, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
        )
        out_pt = config.MODELS_DIR / "fatigue_temporal_cnn.pt"
        save_model(model, mean, std, out_pt)
        print(f"Saved {out_pt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
