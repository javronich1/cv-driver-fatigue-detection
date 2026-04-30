"""Stage 2E / 3C: end-to-end evaluation of the gesture pipeline.

Runs the full pipeline on every sequence-level clip:

    correct sequence  / wrong sequence
    too slow          / incomplete
    random hands      / no hands     (sanity-check distractors)

Supported classifiers:

    --model svm   -> outputs/models/gesture_svm.joblib   (classical, default)
    --model rf    -> outputs/models/gesture_rf.joblib    (classical)
    --model cnn   -> outputs/models/gesture_cnn.pt       (modern, MobileNetV3)

Outputs:
    outputs/reports/gesture_sequence_eval_<model>.csv
    outputs/reports/gesture_sequence_eval_<model>.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                      # noqa: E402
from src.gestures.evaluate_sequences import (                               # noqa: E402
    evaluate_all, make_classical_predictor, make_cnn_predictor, summarise,
)
from src.gestures.state_machine import StateMachineConfig                   # noqa: E402


def _build_predictor(kind: str):
    if kind in ("svm", "rf"):
        from src.gestures.classical import load_model
        path = config.MODELS_DIR / (
            "gesture_svm.joblib" if kind == "svm" else "gesture_rf.joblib"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"missing classifier {path}. "
                f"Run scripts/train_gesture_classical.py first."
            )
        print(f"Loading classical model: {path}")
        return make_classical_predictor(load_model(path)), path

    if kind == "cnn":
        from src.gestures.cnn import load_model
        path = config.MODELS_DIR / "gesture_cnn.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"missing CNN model {path}. "
                f"Run scripts/train_gesture_cnn.py first."
            )
        print(f"Loading CNN model: {path}")
        model, classes, input_size = load_model(path)
        return make_cnn_predictor(model, classes, input_size), path

    raise ValueError(f"Unknown --model {kind!r}; expected svm/rf/cnn.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["svm", "rf", "cnn"], default="svm",
        help="Which classifier to evaluate (default: svm).",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--min-consecutive", type=int, default=4)
    args = parser.parse_args(argv)

    config.ensure_dirs()
    try:
        predictor, model_path = _build_predictor(args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    sm_cfg = StateMachineConfig(
        window_s=args.window_s,
        min_confidence=args.min_confidence,
        min_consecutive=args.min_consecutive,
    )
    print("State machine config:", sm_cfg)

    df = evaluate_all(predictor, sm_cfg, stride=args.stride)
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
