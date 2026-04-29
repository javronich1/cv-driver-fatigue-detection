"""Stage 2B: extract per-frame hand-landmark features for the gesture
training videos and save to disk.

Outputs:
    data_processed/gestures/features.csv

Run from the project root:

    python scripts/extract_gesture_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                 # noqa: E402
from src.gestures.extraction import build_gesture_feature_table        # noqa: E402


def main() -> int:
    config.ensure_dirs()
    out_dir = config.PROCESSED_DIR / "gestures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.csv"

    print(f"Extracting features → {out_path}")
    df = build_gesture_feature_table(stride=5, resize=(540, 960), min_hand_score=0.5)

    df.to_csv(out_path, index=False)
    print()
    print("Rows         :", len(df))
    print("Columns      :", len(df.columns))
    print("Label counts :")
    print(df["label"].value_counts().to_string())
    print()
    print("Per-person breakdown:")
    print(df.groupby(["label", "person"]).size().unstack(fill_value=0))
    print()
    print("Saved to", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
