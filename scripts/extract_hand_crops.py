"""Stage 3A: build a hand-crop dataset for the modern (CNN) gesture
classifier.

Outputs:
    data_processed/hand_crops/{open_palm, thumbs_up, negative}/*.png
    data_processed/hand_crops/index.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                 # noqa: E402
from src.gestures.crops import (                                       # noqa: E402
    CROP_DIR, CROP_INDEX, DEFAULT_CROP_SIZE, DEFAULT_MARGIN,
    build_hand_crop_dataset, save_index,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5,
                        help="Sample every Nth frame (default 5).")
    parser.add_argument("--resize", type=int, nargs=2, default=(540, 960),
                        metavar=("H", "W"),
                        help="Resize frames before detection (default 540 960).")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE,
                        help="Output crop side in pixels (default 96).")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="Bounding-box margin fraction (default 0.30).")
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--no-save", action="store_true",
                        help="Build the index only — useful for a dry run.")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    print(f"Writing hand crops to {CROP_DIR}")
    df = build_hand_crop_dataset(
        stride=args.stride,
        resize=tuple(args.resize),
        out_size=args.crop_size,
        margin=args.margin,
        min_hand_score=args.min_hand_score,
        save_images=not args.no_save,
    )
    save_index(df)
    print(f"\nWrote {len(df)} crops; index → {CROP_INDEX}")
    print(df.groupby(["label", "person"]).size().unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
