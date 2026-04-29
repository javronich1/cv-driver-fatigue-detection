"""Stage 2E: end-to-end evaluation of the classical gesture pipeline.

Loads the trained SVM (default; configurable to RF) and runs the full
pipeline on every sequence-level clip:

    correct sequence  / wrong sequence
    too slow          / incomplete
    random hands      / no hands     (sanity-check distractors)

Outputs:
    outputs/reports/gesture_sequence_eval.csv
    outputs/reports/gesture_sequence_eval.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                      # noqa: E402
from src.gestures.classical import load_model                               # noqa: E402
from src.gestures.evaluate_sequences import evaluate_all, summarise         # noqa: E402
from src.gestures.state_machine import StateMachineConfig                   # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["svm", "rf"], default="svm",
        help="Which classifier to evaluate (default: svm).",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--min-consecutive", type=int, default=4)
    args = parser.parse_args(argv)

    config.ensure_dirs()
    model_path = config.MODELS_DIR / (
        "gesture_svm.joblib" if args.model == "svm" else "gesture_rf.joblib"
    )
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        print("       Run scripts/train_gesture_classical.py first.")
        return 1

    print(f"Loading classifier: {model_path}")
    model = load_model(model_path)

    sm_cfg = StateMachineConfig(
        window_s=args.window_s,
        min_confidence=args.min_confidence,
        min_consecutive=args.min_consecutive,
    )
    print("State machine config:", sm_cfg)

    df = evaluate_all(model, sm_cfg, stride=args.stride)
    summary = summarise(df)
    print(summary)

    out_csv = config.REPORTS_DIR / f"gesture_sequence_eval_{args.model}.csv"
    out_txt = config.REPORTS_DIR / f"gesture_sequence_eval_{args.model}.txt"
    df.to_csv(out_csv, index=False)
    out_txt.write_text(summary)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
