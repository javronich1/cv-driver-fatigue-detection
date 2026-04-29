"""Inventory the local dataset.

Walks every video, probes its metadata with OpenCV, and returns a tidy
DataFrame plus a human-readable summary. This is our Stage 1 sanity check:
if any video can't be read, we want to know *now*, not during training.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from .loader import VideoRecord, discover_all
from ..utils.video import probe


def build_inventory(
    records: List[VideoRecord] | None = None,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Probe every video and return one row per file.

    Columns: dataset, folder_label, canonical_label, person, filename, path,
    width, height, fps, n_frames, duration_s, fourcc, readable, error.
    """
    if records is None:
        records = discover_all()

    rows = []
    iterator = tqdm(records, desc="Probing videos") if show_progress else records
    for rec in iterator:
        row = rec.to_dict()
        info = probe(rec.path)
        if info is None:
            row.update({
                "width": 0, "height": 0, "fps": 0.0,
                "n_frames": 0, "duration_s": 0.0,
                "fourcc": "", "readable": False,
                "error": "cv2.VideoCapture failed to open",
            })
        else:
            row.update({
                "width": info.width,
                "height": info.height,
                "fps": round(info.fps, 3),
                "n_frames": info.n_frames,
                "duration_s": round(info.duration_s, 3),
                "fourcc": info.fourcc,
                "readable": info.readable,
                "error": "" if info.readable else "first frame unreadable",
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summarise(df: pd.DataFrame) -> str:
    """Return a human-readable summary of an inventory dataframe."""
    if df.empty:
        return "No videos found. Is the dataset/ folder populated?"

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("DATASET INVENTORY SUMMARY")
    lines.append("=" * 72)

    total = len(df)
    readable = int(df["readable"].sum())
    bad = total - readable
    total_dur_min = df["duration_s"].sum() / 60.0

    lines.append(f"Total videos       : {total}")
    lines.append(f"Readable           : {readable}")
    lines.append(f"Failed to open/read: {bad}")
    lines.append(f"Total duration     : {total_dur_min:.1f} min")
    lines.append("")

    # Per dataset
    for ds, sub in df.groupby("dataset"):
        lines.append(f"[{ds}]")
        lines.append(f"  videos   : {len(sub)}")
        lines.append(f"  duration : {sub['duration_s'].sum() / 60.0:.1f} min")
        lines.append(f"  persons  : {sorted(sub['person'].dropna().unique())}")
        lines.append("")

    # Per (dataset, label) breakdown
    lines.append("Breakdown by class:")
    g = (
        df.groupby(["dataset", "canonical_label", "person"])
          .agg(videos=("filename", "count"),
               duration_s=("duration_s", "sum"))
          .reset_index()
          .sort_values(["dataset", "canonical_label", "person"])
    )
    for _, row in g.iterrows():
        lines.append(
            f"  {row['dataset']:>9} / {row['canonical_label']:<18}"
            f" / {str(row['person']):<8} "
            f" {int(row['videos']):>3} clips, {row['duration_s']/60:>5.2f} min"
        )

    # Resolution/fps consistency
    lines.append("")
    lines.append("Resolution / fps distribution:")
    res_df = (
        df[df["readable"]]
        .groupby(["width", "height", "fps"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    for _, row in res_df.iterrows():
        lines.append(
            f"  {row['width']}x{row['height']} @ {row['fps']:.2f} fps : "
            f"{int(row['count'])} videos"
        )

    if bad:
        lines.append("")
        lines.append("BAD FILES:")
        for _, row in df[~df["readable"]].iterrows():
            lines.append(f"  {row['path']}  -- {row['error']}")

    lines.append("=" * 72)
    return "\n".join(lines)


def save_inventory(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Save the dataframe to CSV + a human-readable summary to TXT."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inventory.csv"
    txt_path = out_dir / "inventory_summary.txt"
    df.to_csv(csv_path, index=False)
    txt_path.write_text(summarise(df))
    return csv_path, txt_path
