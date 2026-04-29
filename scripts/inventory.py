"""Run the dataset inventory and save the report.

Usage:
    python scripts/inventory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                      # noqa: E402
from src.data.inventory import build_inventory, save_inventory, summarise  # noqa: E402


def main() -> int:
    config.ensure_dirs()
    if not config.DATASET_DIR.is_dir():
        print(f"ERROR: dataset folder not found at {config.DATASET_DIR}")
        return 1

    df = build_inventory()
    if df.empty:
        print("No videos found.")
        return 1

    summary = summarise(df)
    print(summary)

    csv_path, txt_path = save_inventory(df, config.REPORTS_DIR)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {txt_path}")

    n_bad = int((~df["readable"]).sum())
    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
