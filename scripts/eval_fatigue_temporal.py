"""Stage 4D: per-clip fatigue evaluation with temporal aggregation.

Inputs:
    outputs/reports/fatigue_features.csv

Outputs:
    outputs/reports/fatigue_clip_eval_<svm|rf>.csv   (one row per clip)
    outputs/reports/fatigue_clip_eval.txt
    outputs/figures/fatigue_clip_confusion_<svm|rf>_<method>_test=person*.png
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
from src.fatigue.classical import (                                     # noqa: E402
    CLASSES, make_random_forest, make_svm,
)
from src.fatigue.temporal_eval import evaluate_clip_level_loso          # noqa: E402
from src.utils.plotting import plot_confusion                           # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",
                        default=str(config.REPORTS_DIR / "fatigue_features.csv"))
    parser.add_argument("--window", type=int, default=5,
                        help="Sliding-window size in frames (default 5).")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: missing {feat_path}. Run extract_fatigue_features.py.")
        return 1
    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} face-frames from {feat_path}")

    summary_lines = ["=" * 72, "FATIGUE — PER-CLIP TEMPORAL EVAL", "=" * 72]
    summary_lines.append(f"Window size  : {args.window} frames")
    summary_lines.append(f"Total clips  : {df['video'].nunique()}")

    for short, factory in (("svm", make_svm), ("rf", make_random_forest)):
        results, clip_preds = evaluate_clip_level_loso(
            df, factory, window=args.window,
        )
        summary_lines.append("\n" + "-" * 72)
        summary_lines.append(f"Model: {short.upper()}")
        summary_lines.append("-" * 72)

        # Per fold + method.
        accs = {"mean_prob": [], "window_vote": []}
        f1s = {"mean_prob": [], "window_vote": []}
        for r in results:
            accs[r.method].append(r.accuracy)
            f1s[r.method].append(r.macro_f1)
            summary_lines.append(
                f"\n[{r.fold_name}] method={r.method}  "
                f"accuracy={r.accuracy:.3f}  macro_F1={r.macro_f1:.3f}"
            )
            summary_lines.append(r.report)
            summary_lines.append(
                f"confusion (rows=true {CLASSES}, cols=pred):"
            )
            summary_lines.append(np.array2string(r.confusion))
            plot_confusion(
                cm=r.confusion,
                classes=list(CLASSES),
                title=f"Fatigue {short} ({r.method}) — {r.fold_name}  "
                      f"(macro-F1={r.macro_f1:.3f})",
                out_path=config.FIGURES_DIR
                         / f"fatigue_clip_confusion_{short}_{r.method}_{r.fold_name}.png",
                normalize=False,
            )

        for method in ("mean_prob", "window_vote"):
            if accs[method]:
                summary_lines.append(
                    f"\n  Mean ({short}, {method}): "
                    f"accuracy={np.mean(accs[method]):.3f}  "
                    f"macro_F1={np.mean(f1s[method]):.3f}"
                )

        # Save per-clip predictions for downstream figures.
        out_csv = config.REPORTS_DIR / f"fatigue_clip_eval_{short}.csv"
        clip_preds.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

    out_txt = config.REPORTS_DIR / "fatigue_clip_eval.txt"
    out_txt.write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"\nWrote {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
