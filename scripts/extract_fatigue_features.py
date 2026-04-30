"""Stage 4B: build the per-frame fatigue feature dataframe.

Outputs:
    outputs/reports/fatigue_features.csv

Usage:
    python scripts/extract_fatigue_features.py [--stride 5]
                                                [--include-transition]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                                    # noqa: E402
from src.fatigue.extraction import (                                      # noqa: E402
    build_fatigue_feature_table, save_feature_table,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5,
                        help="Sample every Nth frame (default 5).")
    parser.add_argument("--resize", type=int, nargs=2, default=(540, 960),
                        metavar=("H", "W"),
                        help="Resize frames before detection (default 540 960).")
    parser.add_argument("--include-transition", action="store_true",
                        help="Keep the 'transition' coarse class.")
    parser.add_argument("--keep-missing-face", action="store_true",
                        help="Don't drop frames where MediaPipe found no face.")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    df = build_fatigue_feature_table(
        stride=args.stride,
        resize=tuple(args.resize),
        drop_missing_face=not args.keep_missing_face,
        include_transition=args.include_transition,
    )

    out = config.REPORTS_DIR / "fatigue_features.csv"
    save_feature_table(df, out)

    print(f"\nWrote {len(df)} rows -> {out}")
    if "coarse_label" in df.columns and "person" in df.columns:
        print("\nrows by coarse_label x person:")
        print(df.groupby(["coarse_label", "person"]).size()
                .unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
